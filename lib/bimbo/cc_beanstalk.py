#!/usr/bin/env python

import os
import sys
import json
import zlib
import time
import glob
import click

import pandas as pd
import numpy as np

import json

import pymongo
import beanstalkc

from utils import log
from utils import DEBUG, INFO, WARN
from bimbo.constants import get_stats_mongo_collection, load_median_solution
from bimbo.constants import COMPETITION_CC_NAME, IP_BEANSTALK, PORT_BEANSTALK, TIMEOUT_BEANSTALK, MONGODB_URL, MONGODB_DATABASE, MONGODB_COLUMNS, MONGODB_STATS_CC_COLLECTION
from bimbo.constants import COLUMN_AGENCY, COLUMN_CHANNEL, COLUMN_ROUTE, COLUMN_PRODUCT, COLUMN_CLIENT, COLUMNS, SPLIT_PATH
from bimbo.constants import NON_PREDICTABLE

def cc_calculation(product_id, history, threshold_value=0, progress_prefix=None, alternative_filetype="", alternative_id=""):
    alternative_solution = load_median_solution(alternative_filetype)

    loss_median_sum = 0
    loss_cc_sum, loss_count = 0, 0.00000001

    rtype = NON_PREDICTABLE
    record = None
    for client_id, values in history.items():
        prediction_median = max(0, np.median(history[client_id][:-1]))
        prediction_cc = prediction_median

        client_mean = np.mean(values[1:-1])
        if np.sum(values[1:-1]) == 0:
            rtype = "median"
            prediction_cc = alternative_solution.get("{}_{}".format(alternative_id, product_id), 0)

            record = {"product_id": product_id, "client_id": int(client_id), "history": values, "cc": [], "prediction": NON_PREDICTABLE}
        else:
            cc_client_ids, cc_matrix = [client_id], [values[1:-1]]
            for cc_client_id, cc_client_values in history.items():
                if client_id != cc_client_id:
                    cc_client_ids.append(cc_client_id)
                    cc_matrix.append(cc_client_values[0:-2])

            results_cc = {}
            if len(cc_client_ids) > 1:
                cc_results = np.corrcoef(cc_matrix)[0]
                results_cc = dict(zip(cc_client_ids, cc_results))

                num_sum, num_count = 0, 0.00000001
                for c, value in sorted(results_cc.items(), key=lambda (k, v): abs(v), reverse=True):
                    if c == client_id or np.isnan(value):
                        continue
                    elif abs(value) > threshold_value :
                        ratio = client_mean/np.mean(history[c][0:-2])
                        score = (history[c][-2] - history[c][-3])*value*ratio
                        num_sum += score
                        num_count += 1

                    else:
                        break

                if np.sum(np.array(history[client_id]) == 0) > 4:
                    prediction_cc = 0
                else:
                    prediction_cc = max(0, values[-2] + num_sum/num_count)

                prediction_cc = prediction_cc*0.71 + 0.243

                matrix = []
                for cid, value in results_cc.items():
                    if not np.isnan(value):
                        matrix.append({"client_id": int(cid), "value": value})

                rtype = "cc"
                record = {"product_id": product_id, "client_id": int(client_id), "history": values, "cc": matrix, "prediction": prediction_cc}

        loss = (np.log1p(prediction_cc)-np.log1p(values[-1]))**2
        #loss = (np.log1p(prediction_cc)-values[-1])**2

        loss_count += 1
        loss_cc_sum += loss

        loss_median_sum += (np.log1p(prediction_median)-np.log1p(values[-1]))**2
        #loss_median_sum += (prediction_median-values[-1])**2

        log("{} {}/{} >>> {} - {} - {} - {:4f}({:4f}) - {:4f}({:4f})".format(\
            "{}. {}/{}".format(rtype, progress_prefix[0], progress_prefix[1]) if progress_prefix else "",
            int(loss_count), len(history), product_id, client_id, history[client_id], prediction_cc, prediction_median, np.sqrt(loss_cc_sum/loss_count), np.sqrt(loss_median_sum/loss_count)), INFO)

        yield rtype, record, loss

    log("RMLSE for {}: {:6f}".format(product_id, np.sqrt(loss_cc_sum/loss_count)), INFO)

    yield "stats", product_id, np.sqrt(loss_cc_sum/loss_count)

def consumer(task=COMPETITION_CC_NAME):
    client = pymongo.MongoClient(MONGODB_URL)

    talk = beanstalkc.Connection(host=IP_BEANSTALK, port=PORT_BEANSTALK)
    talk.watch(task)

    loss_sum, loss_count = 0, 0
    while True:
        try:
            job = talk.reserve(timeout=TIMEOUT_BEANSTALK)
            if job:
                o = json.loads(zlib.decompress(job.body))
                mongodb_database, mongodb_collection = o["mongodb_database"], o["mongodb_collection"]
                client[mongodb_database][MONGODB_STATS_CC_COLLECTION].create_index("product_id")
                client[mongodb_database][MONGODB_STATS_CC_COLLECTION].create_index("client_id")

                product_id, history = o["product_id"], o["history"]

                first_client_id = int(history.keys()[0])
                count = client[mongodb_database][MONGODB_STATS_CC_COLLECTION].count({"product_id": product_id, "client_id": first_client_id})
                if count == 0:
                    timestamp_start = time.time()
                    count, records = 0, []
                    for rtype, record, loss_cc_sum in cc_calculation(product_id, history):
                        if rtype in [NON_PREDICTABLE, "record"]:
                            records.append(record)

                            if rtype == "record":
                                loss_sum += loss_cc_sum
                                loss_count += 1
                        elif rtype == "stats":
                            client[mongodb_database][MONGODB_STATS_CC_COLLECTION].insert({"product_id": record, "rmlse": loss_cc_sum})

                    if records:
                        client[mongodb_database][mongodb_collection].insert_many(records)
                        count += len(records)

                    log("Current RMLSE: {:8f}".format(np.sqrt(loss_sum/loss_count)), INFO)

                    timestamp_end = time.time()
                    log("Cost {:4f} secends to insert {} records into the mongodb({}-{})".format(timestamp_end-timestamp_start, count, mongodb_database, mongodb_collection), INFO)

                job.delete()
        except BeanstalkcException as e:
            log("Error occurs, {}".format(e), WARN)
        except Exception as e:
            raise

    client.close()
    talk.close()

def producer(filetype, task=COMPETITION_CC_NAME, ttr=TIMEOUT_BEANSTALK):
    filepath = os.path.join(SPLIT_PATH, COLUMNS[filetype[0]], "train", "{}.csv".format(filetype[1]))

    talk = beanstalkc.Connection(host=IP_BEANSTALK, port=PORT_BEANSTALK)
    talk.watch(task)

    shift_week = 3

    def all_zero_list():
        return [0 for _ in range(3, 10)]

    history = {}
    with open(filepath, "rb") as INPUT:
        header = True

        for line in INPUT:
            if header:
                header = False
                continue

            week, agency_id, channel_id, route_id, client_id, product_id, sales_unit, sales_price, return_unit, return_price, prediction_unit = line.strip().split(",")

            week = int(week)
            client_id = int(client_id)
            product_id = int(product_id)
            prediction_unit = int(prediction_unit)

            history.setdefault(product_id, {})
            history[product_id].setdefault(client_id, all_zero_list())
            history[product_id][client_id][week-shift_week] = prediction_unit

    for product_id, info in history.items():
        request = {"product_id": product_id,
                   "mongodb_database": "cc_{}_{}".format(filetype[0], filetype[1]),
                   "mongodb_collection": get_stats_mongo_collection("{}".format(MONGODB_COLUMNS[COLUMN_PRODUCT])),
                   "history": info}

        talk.put(zlib.compress(json.dumps(request)), ttr=TIMEOUT_BEANSTALK)
        log("Put request(product_id={} from {}) into the queue".format(product_id, os.path.basename(filepath)), INFO)

    talk.close()

@click.command()
@click.option("--mode", required=True, help="producer|consumer")
@click.option("--column", default=None, help="split column")
def cc(mode, column):
    if mode.lower() == "producer":
        pattern_file = os.path.join(SPLIT_PATH, COLUMNS[column], "train", "*.csv")
        for filepath in glob.iglob(pattern_file):
            filename = os.path.basename(filepath)
            fid = filename.replace(".csv", "")

            producer((column, fid))
    elif mode.lower() == "consumer":
        consumer()

if __name__ == "__main__":
    cc()

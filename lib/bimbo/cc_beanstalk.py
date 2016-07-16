#!/usr/bin/env python

import os
import json
import zlib
import time

import threading
import Queue

import pandas as pd
import numpy as np

import pymongo
import beanstalkc

from utils import log
from utils import DEBUG, INFO, WARN
from bimbo.constants import get_stats_mongo_collection, load_median_route_solution, get_median
from bimbo.constants import COMPETITION_CC_NAME, IP_BEANSTALK, PORT_BEANSTALK, TIMEOUT_BEANSTALK, MONGODB_URL, MONGODB_DATABASE, MONGODB_COLUMNS, MONGODB_STATS_CC_COLLECTION
from bimbo.constants import COLUMN_AGENCY, COLUMN_CHANNEL, COLUMN_ROUTE, COLUMN_PRODUCT, COLUMN_CLIENT, COLUMNS, SPLIT_PATH
from bimbo.constants import NON_PREDICTABLE

def cc_calculation(week, filetype, product_id, history, threshold_value=0, progress_prefix=None, median_solution=([], [])):
    end_idx = 7 - (9-week)

    loss_median_sum = 0
    loss_cc_sum, loss_count = 0, 0.00000001

    rtype = NON_PREDICTABLE
    record = None
    for client_id, values in history.items():
        prediction_median = get_median(median_solution[0], median_solution[1], {filetype[0]: filetype[1], COLUMN_PRODUCT: product_id, COLUMN_CLIENT: client_id})
        prediction_cc = prediction_median

        client_mean = np.mean(values[1:end_idx])
        if np.sum(values[1:end_idx-1]) == 0:
            rtype = "median"

            record = {"product_id": product_id, "client_id": int(client_id), "history": values, "cc": [], "prediction": NON_PREDICTABLE}
        else:
            cc_client_ids, cc_matrix = [client_id], [values[1:end_idx]]
            for cc_client_id, cc_client_values in history.items():
                if client_id != cc_client_id:
                    cc_client_ids.append(cc_client_id)
                    cc_matrix.append(cc_client_values[0:end_idx-1])

            results_cc = {}
            if len(cc_client_ids) > 1:
                cc_results = np.corrcoef(cc_matrix)[0]
                results_cc = dict(zip(cc_client_ids, cc_results))

                num_sum, num_count = 0, 0.00000001
                for c, value in sorted(results_cc.items(), key=lambda (k, v): abs(v), reverse=True):
                    if c == client_id or np.isnan(value):
                        continue
                    elif abs(value) > threshold_value :
                        ratio = client_mean/np.mean(history[c][0:end_idx-1])
                        score = (history[c][end_idx-2] - history[c][end_idx-3])*value*ratio
                        num_sum += score
                        num_count += 1

                    else:
                        break

                if np.sum(np.array(history[client_id]) == 0) > 4:
                    prediction_cc = 0
                else:
                    prediction_cc = max(0, values[end_idx-2] + num_sum/num_count)

                prediction_cc = prediction_cc*0.71 + 0.243

                matrix = []
                for cid, value in results_cc.items():
                    if not np.isnan(value):
                        matrix.append({"client_id": int(cid), "value": value})

                rtype = "cc"
                record = {"product_id": product_id, "client_id": int(client_id), "history": values, "cc": matrix, "prediction": prediction_cc}

        loss = (np.log1p(prediction_cc)-np.log1p(values[-1]))**2

        loss_count += 1
        loss_cc_sum += loss

        loss_median_sum += (np.log1p(prediction_median)-np.log1p(values[-1]))**2

        log("{} {}. {}/{} >>> {} - {} - {} - {:4f}({:4f}) - {:4f}({:4f})".format(\
            "{}/{}".format(progress_prefix[0], progress_prefix[1]) if progress_prefix else "",
            rtype, int(loss_count), len(history), product_id, client_id, history[client_id], prediction_cc, prediction_median, np.sqrt(loss_cc_sum/loss_count), np.sqrt(loss_median_sum/loss_count)), INFO)

        yield rtype, record, loss

    log("RMLSE for {}: {:6f}".format(product_id, np.sqrt(loss_cc_sum/loss_count)), INFO)

    yield "stats", product_id, np.sqrt(loss_cc_sum/loss_count)

def consumer(median_solution, task=COMPETITION_CC_NAME):
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
                week, filetype = o["week"], o["filetype"]

                client[mongodb_database][mongodb_collection].create_index([("client_id", pymongo.ASCENDING), ("product_id", pymongo.ASCENDING)])
                client[mongodb_database][MONGODB_STATS_CC_COLLECTION].create_index("product_id")
                client[mongodb_database][MONGODB_STATS_CC_COLLECTION].create_index("client_id")

                product_id, history = o["product_id"], o["history"]

                timestamp_start = time.time()

                for rtype, record, loss_cc_sum in cc_calculation(week, (COLUMNS[filetype[0]], filetype[1]), product_id, history, median_solution=median_solution):
                    if rtype in ["cc", "median"]:
                        loss_sum += loss_cc_sum
                        loss_count += 1

                        log(client[mongodb_database][mongodb_collection].update({"client_id": record["client_id"], "product_id": record["product_id"]}, {"$set": record}, upsert=True), INFO)
                    elif rtype == "stats":
                        log(client[mongodb_database][MONGODB_STATS_CC_COLLECTION].insert({"product_id": record, "rmlse": loss_cc_sum}), INFO)

                log("Current RMLSE: {:8f} for {} records".format(np.sqrt(loss_sum/loss_count), loss_count), INFO)

                timestamp_end = time.time()

                job.delete()
        except beanstalkc.BeanstalkcException as e:
            log("Error occurs, {}".format(e), WARN)
        except Exception as e:
            raise

    client.close()
    talk.close()

def get_history(filepath, shift_week=3, week=[3, 10]):
    history = {}
    with open(filepath, "rb") as INPUT:
        header = True

        for line in INPUT:
            if header:
                header = False
                continue

            w, agency_id, channel_id, route_id, client_id, product_id, sales_unit, sales_price, return_unit, return_price, prediction_unit = line.strip().split(",")

            w = int(w)
            client_id = int(client_id)
            product_id = int(product_id)
            prediction_unit = int(prediction_unit)

            history.setdefault(product_id, {})
            history[product_id].setdefault(client_id, [0 for _ in range(week[0], week[1])])
            history[product_id][client_id][w-shift_week] = prediction_unit

    return history

def producer(week, filetype, task=COMPETITION_CC_NAME, ttr=TIMEOUT_BEANSTALK):
    filepath = os.path.join(SPLIT_PATH, COLUMNS[filetype[0]], "train", "{}.csv".format(filetype[1]))

    talk = beanstalkc.Connection(host=IP_BEANSTALK, port=PORT_BEANSTALK)
    talk.watch(task)

    history = get_history(filepath)

    for product_id, info in history.items():
        request = {"product_id": product_id,
                   "week": week,
                   "filetype": list(filetype),
                   "mongodb_database": "cc_{}_{}".format(filetype[0], filetype[1]),
                   "mongodb_collection": get_stats_mongo_collection("{}".format(MONGODB_COLUMNS[COLUMN_PRODUCT])),
                   "history": info}

        talk.put(zlib.compress(json.dumps(request)), ttr=TIMEOUT_BEANSTALK)
        log("Put request(product_id={} from {}) into the {}".format(product_id, os.path.basename(filepath), task), INFO)

    talk.close()

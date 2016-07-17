#!/usr/bin/env python

import os
import json
import zlib
import time

import threading

import pandas as pd
import numpy as np

import pymongo
import beanstalkc

from utils import log
from utils import DEBUG, INFO, WARN
from bimbo.constants import get_mongo_connection, get_cc_mongo_collection, get_prediction_mongo_collection, load_median_route_solution, get_median
from bimbo.constants import COMPETITION_CC_NAME, IP_BEANSTALK, PORT_BEANSTALK, TIMEOUT_BEANSTALK
from bimbo.constants import MONGODB_PREDICTION_COLLECTION, MONGODB_STATS_CC_COLLECTION, MONGODB_PREDICTION_DATABASE, MONGODB_CC_DATABASE
from bimbo.constants import COLUMN_AGENCY, COLUMN_CHANNEL, COLUMN_ROUTE, COLUMN_PRODUCT, COLUMN_CLIENT, COLUMNS, MONGODB_COLUMNS, SPLIT_PATH
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

        record = {"groupby": MONGODB_COLUMNS[filetype[0]], "week": week, "product_id": product_id, "client_id": int(client_id), "history": values, "cc": []}
        prediction = {"groupby": MONGODB_COLUMNS[filetype[0]], "week": week, "client_id": int(client_id), "product_id": product_id, "prediction_type": None, "prediction": NON_PREDICTABLE}

        client_mean = np.mean(values[1:end_idx])
        if np.sum(values[1:end_idx-1]) == 0:
            rtype = "median"
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
                record["cc"] = matrix
                prediction["prediction"] = prediction_cc

        loss = (np.log1p(prediction_cc)-np.log1p(values[-1]))**2

        loss_count += 1
        loss_cc_sum += loss

        loss_median_sum += (np.log1p(prediction_median)-np.log1p(values[-1]))**2

        log("{} {}. {}/{} >>> {} - {} - {} - {:4f}({:4f}) - {:4f}({:4f})".format(\
            "{}/{}".format(progress_prefix[0], progress_prefix[1]) if progress_prefix else "",
            rtype, int(loss_count), len(history), product_id, client_id, history[client_id], prediction_cc, prediction_median, np.sqrt(loss_cc_sum/loss_count), np.sqrt(loss_median_sum/loss_count)), DEBUG)

        prediction["prediction_type"] = rtype

        yield rtype, (record, prediction), loss

    log("RMLSE for {} -  {}/{} = {:6f}".format(product_id, loss_cc_sum, loss_count, np.sqrt(loss_cc_sum/loss_count)), INFO)

    yield "stats", product_id, np.sqrt(loss_cc_sum/loss_count)

def consumer(median_solution, task=COMPETITION_CC_NAME, n_jobs=4):
    client = get_mongo_connection()

    talk = beanstalkc.Connection(host=IP_BEANSTALK, port=PORT_BEANSTALK)
    talk.watch(task)

    loss_sum, loss_count = 0, 0.00000001
    while True:
        try:
            job = talk.reserve(timeout=TIMEOUT_BEANSTALK)
            if job:
                o = json.loads(zlib.decompress(job.body))
                week, filetype = o["week"], o["filetype"]

                mongodb_cc_database, mongodb_cc_collection = o["mongodb_cc"]
                mongodb_prediction_database, mongodb_prediction_collection = o["mongodb_prediction"]
                mongodb_stats_database, mongodb_stats_collection = o["mongodb_stats"]

                product_id, history = o["product_id"], o["history"]

                timestamp_start = time.time()

                for rtype, record, loss_cc_sum in cc_calculation(week, (COLUMNS[filetype[0]], filetype[1]), product_id, history, median_solution=median_solution):
                    if rtype in ["cc", "median"]:
                        loss_sum += loss_cc_sum
                        loss_count += 1

                        cc, prediction = record

                        query = {"week": week, "groupby": cc["groupby"], "client_id": cc["client_id"], "product_id": cc["product_id"]}
                        client[mongodb_cc_database][mongodb_cc_collection].update(query, {"$set": cc}, upsert=True)

                        client[mongodb_prediction_database][mongodb_prediction_collection].update(query, {"$set": prediction}, upsert=True)
                    elif rtype == "stats":
                        query = {"week": week, "groupby": filetype[0], "product_id": record}
                        r = {"week": week, "groupby": filetype[0], "product_id": record, "rmlse": loss_cc_sum}

                        client[mongodb_stats_database][mongodb_stats_collection].update(query, {"$set": r}, upsert=True)

                log("Current RMLSE: {:8f} for {} records".format(np.sqrt(loss_sum/loss_count), loss_count), INFO)

                timestamp_end = time.time()

                job.delete()
        except beanstalkc.BeanstalkcException as e:
            log("Error occurs, {}".format(e), WARN)
        except Exception as e:
            raise

    talk.close()
    client.close()

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

    client = get_mongo_connection()

    mongodb_cc_database, mongodb_cc_collection = MONGODB_CC_DATABASE, get_cc_mongo_collection(MONGODB_COLUMNS[COLUMN_PRODUCT])
    client[mongodb_cc_database][mongodb_cc_collection].create_index([("week", pymongo.ASCENDING), ("groupby", pymongo.ASCENDING), ("client_id", pymongo.ASCENDING), ("product_id", pymongo.ASCENDING)])
    client[mongodb_cc_database][mongodb_cc_collection].create_index([("week", pymongo.ASCENDING), ("groupby", pymongo.ASCENDING)])
    client[mongodb_cc_database][mongodb_cc_collection].create_index("week")
    client[mongodb_cc_database][mongodb_cc_collection].create_index("groupby")
    client[mongodb_cc_database][mongodb_cc_collection].create_index("product_id")
    client[mongodb_cc_database][mongodb_cc_collection].create_index("client_id")

    mongodb_prediction_database, mongodb_prediction_collection = MONGODB_PREDICTION_DATABASE, get_prediction_mongo_collection(MONGODB_COLUMNS[COLUMN_PRODUCT])
    client[mongodb_prediction_database][mongodb_prediction_collection].create_index([("week", pymongo.ASCENDING),
                                                                                     ("groupby", pymongo.ASCENDING),
                                                                                     ("client_id", pymongo.ASCENDING),
                                                                                     ("product_id", pymongo.ASCENDING)])

    mongodb_stats_database, mongodb_stats_collection = MONGODB_CC_DATABASE, MONGODB_STATS_CC_COLLECTION
    client[mongodb_stats_database][mongodb_stats_collection].create_index([("week", pymongo.ASCENDING), ("groupby", pymongo.ASCENDING), ("product_id", pymongo.ASCENDING)])

    history = get_history(filepath)
    for product_id, info in history.items():
        request = {"product_id": product_id,
                   "week": week,
                   "filetype": list(filetype),
                   "mongodb_cc": [MONGODB_CC_DATABASE, get_cc_mongo_collection("{}_{}".format("_".join(filetype), MONGODB_COLUMNS[COLUMN_PRODUCT]))],
                   "mongodb_prediction": [MONGODB_PREDICTION_DATABASE, get_prediction_mongo_collection("{}_{}".format("_".join(filetype), MONGODB_COLUMNS[COLUMN_PRODUCT]))],
                   "mongodb_stats": [MONGODB_CC_DATABASE, MONGODB_STATS_CC_COLLECTION],
                   "history": info}

        talk.put(zlib.compress(json.dumps(request)), ttr=TIMEOUT_BEANSTALK)
        log("Put request(product_id={} from {}) into the {}".format(product_id, os.path.basename(filepath), task), INFO)

    client.close()
    talk.close()

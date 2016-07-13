#!/usr/bin/env python

import os
import sys
import time
import glob

import pandas as pd
import numpy as np

import json

import pymongo
import beanstalkc

from utils import log
from utils import DEBUG, INFO, WARN
from bimbo.constants import get_stats_mongo_collection
from bimbo.constants import COMPETITION_CC_NAMCOMPETITION_CC_NAMEE, IP_BEANSTALK, PORT_BEANSTALK, TIMEOUT_BEANSTALK, MONGODB_URL, MONGODB_DATABASE, MONGODB_COLUMNS, BATCH_JOB
from bimbo.constants import COLUMN_AGENCY, COLUMN_CHANNEL, COLUMN_ROUTE, COLUMN_PRODUCT, COLUMN_CLIENT, COLUMNS, SPLIT_PATH

BATCH_JOB = 5000

def cc_calculation(product_id, history, threshold_value=0):
    loss_median_sum = 0
    loss_cc_sum, loss_count = 0, 0

    for client_id, values in history.items():
        client_mean = np.mean(values[1:-1])

        cc_client_ids, cc_matrix = [client_id], [values[1:-1]]
        for cc_client_id, cc_client_values in history.items():
            if client_id != cc_client_id:
                cc_client_ids.append(cc_client_id)
                cc_matrix.append(cc_client_values[0:-2])

        prediction_median = np.median(history[client_id][:-1])
        prediction_cc = prediction_median

        cc_results = []
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

        prediction_cc = max(0, values[-2] + num_sum/num_count)

        loss_count += 1
        loss_cc_sum += (np.log(prediction_cc+1)-np.log(values[-1]+1))**2
        loss_median_sum += (np.log(prediction_median+1)-np.log(values[-1]+1))**2

        log("{}/{} >>> {} - {} - {} - {:4f}({:4f}) - {:4f}({:4f})".format(\
            loss_count, len(info)**2, product_id, client_id, history[client_id], prediction_cc, prediction_median, np.sqrt(loss_cc_sum/loss_count), np.sqrt(loss_median_sum/loss_count)), INFO)

        yield {"client_id": client_id, "history": values, "cc": [{"client_id": cid, "cc": cc} for cid, cc in cc_results.items()], "prediction": prediction_cc}

    log("Total RMLSE for {}: {:6f}".format(product_id, np.sqrt(loss_cc_sum/loss_count)), INFO)

def consumer(task=COMPETITION_CC_NAME):
    client = pymongo.MongoClient(MONGODB_URL)

    talk = beanstalkc.Connection(host=IP_BEANSTALK, port=PORT_BEANSTALK)
    talk.watch(task)

    while True:
        job = TALK.reserve(timeout=TIMEOUT_BEANSTALK)
        if job:
            try:
                o = json.loads(job.body)
                mongodb_database, mongodb_collection = o["monogdb_database"], o["mongodb_collection"]
                product_id, history = o["product_id"], o["history"]

                timestamp_start = time.time()
                count, records = 0, []
                for record in cc_calculation(product_id, history):
                    records.append(record)

                    if len(records) > BATCH_JOB:
                        client[mongodb_database][mongodb_collection].insert_many(records)
                        count += len(records)

                        records = []

                if records:
                    client[mongodb_database][mongodb_collection].insert_many(records)
                    count += len(records)

                timestamp_end = time.time()
                log("Cost {:4f} secends to insert {} records into the mongodb({}-{})".format(timestamp_end-timestamp_start, count, mongodb_database, mongodb_collection), INFO)

                job.delete()
            except Exception as e:
                log("Error occurs, {}".format(e), WARN)

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
                   "mongodb_collection": get_stats_mongo_collection("{}_{}".format(MONGODB_COLUMNS[COLUMN_PRODUCT], product_id)),
                   "history": info}

        talk.put(json.dumps(request), ttr=3600)
        log("Put request(product_id={}) into the queue".format(product_id), INFO)

    talk.close()

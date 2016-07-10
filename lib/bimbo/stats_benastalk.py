#!/usr/bin/env python

import os
import sys
import time
import glob

import pandas as pd
import numpy as np

import json
import itertools

import pymongo
import beanstalkc

from utils import log
from utils import DEBUG, INFO, WARN
from bimbo.constants import get_stats_mongo_collection
from bimbo.constants import COMPETITION_NAME, IP_BEANSTALK, PORT_BEANSTALK, TIMEOUT_BEANSTALK, MONGODB_URL, MONGODB_DATABASE, MONGODB_COLUMNS
from bimbo.constants import COLUMN_AGENCY, COLUMN_CHANNEL, COLUMN_ROUTE, COLUMN_PRODUCT, COLUMN_CLIENT, COLUMNS, SPLIT_PATH

def stats(filepath_train, filepath_test, columns, fixed_column):
    df_train = pd.read_csv(filepath_train)
    df_test = pd.read_csv(filepath_test)

    for rid in range(0, df_test.shape[0]):
        timestamp_start = time.time()

        row_id, week_num, agency_id, channel_id, route_id, client_id, product_id = df_test.values[rid]
        record = {
                    "row_id": row_id,
                    "fixed_column": fixed_column,
                    "week_num": week_num,
                    MONGODB_COLUMNS[COLUMN_AGENCY]: agency_id,
                    MONGODB_COLUMNS[COLUMN_CHANNEL]: channel_id,
                    MONGODB_COLUMNS[COLUMN_ROUTE]: route_id,
                    MONGODB_COLUMNS[COLUMN_CLIENT]: client_id,
                    MONGODB_COLUMNS[COLUMN_PRODUCT]: product_id,
                 }

        for num in range(len(columns), 0, -1):
            key = "{}_dimension".format(num)
            record.setdefault(key, [])

            for combination in itertools.combinations(columns, num):
                df = df_train.copy()[df_train[COLUMN_PRODUCT] == product_id]

                dimensions = []
                for criteria in combination:
                    dimensions.append(criteria.lower())
                    df = df[df[criteria] == df_test[criteria].values[rid]]

                    if df.shape[0] == 0:
                        break

                count = df.shape[0]
                if count > 0:
                    record[key].append({"|".join(dimensions): count})

            if record[key]:
                record["matching_count"] = num

                break
            else:
                del record[key]

        yield record

        if rid > 0 and (rid % 1000 == 0):
            timestamp_end = time.time()
            log("Cost {:4f} secends to finish {}/{} records for {}".format(timestamp_end-timestamp_start, rid+1, df_test.shape[0], filepath_test), INFO)

def consumer(task=COMPETITION_NAME):
    CLIENT = pymongo.MongoClient(MONGODB_URL)

    TALK = beanstalkc.Connection(host=IP_BEANSTALK, port=PORT_BEANSTALK)
    TALK.watch(task)

    while True:
        job = TALK.reserve(timeout=TIMEOUT_BEANSTALK)
        if job:
            try:
                o = json.loads(job.body)

                fixed_column = o["fixed_column"]
                filepath_train, filepath_test = o["filepath_train"], o["filepath_test"]

                collection = CLIENT[MONGODB_DATABASE][get_stats_mongo_collection(fixed_column)]

                columns = MONGODB_COLUMNS.keys()[:]
                columns.remove(COLUMN_PRODUCT)

                if COLUMNS[fixed_column] in columns:
                    columns.remove(COLUMNS[fixed_column])

                # create index
                for column in columns + ["product_id"] + ["row_id"]:
                    collection.create_index(MONGODB_COLUMNS[column])

                timestamp_start, timestamp_end = None, None

                records = []
                for record in stats(filepath_train, filepath_test, columns, fixed_column):
                    if timestamp_start == None:
                        timestamp_start = time.time()

                    records.append(record)

                    if len(records) == 5000:
                        collection.insert_many(records)

                        timestamp_end = time.time()
                        log("Cost {:4f} secends to insert {} records into {}-{}".format(timestamp_end-timestamp_start, len(records), MONGODB_DATABASE, get_stats_mongo_collection(fixed_column)), INFO)

                        timestamp_start = None
                        records = []

                if records:
                    collection.insert_many(records)
                    log("Insert {} records into {}-{}".format(len(records), MONGODB_DATABASE, get_stats_mongo_collection(fixed_column)), INFO)

                job.delete()
            except Exception as e:
                log("Error occurs, {}".format(e), WARN)

                raise

    CLIENT.close()
    TALK.close()

def producer(column, is_testing, task=COMPETITION_NAME, ttr=TIMEOUT_BEANSTALK):
    TALK = beanstalkc.Connection(host=IP_BEANSTALK, port=PORT_BEANSTALK)
    TALK.watch(task)

    files = os.path.join(SPLIT_PATH, COLUMNS[column], "test", "*.csv")
    for filepath_test in glob.iglob(files):
        filepath_train = filepath_test.replace("test", "train")

        string = {"fixed_column": column,
                  "filepath_train": filepath_train,
                  "filepath_test": filepath_test}

        TALK.put(json.dumps(string), ttr=ttr)
        log("Put {} into the queue".format(filepath_test), INFO)

        if is_testing:
            break

    TALK.close()

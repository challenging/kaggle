#!/usr/bin/env python

import os
import time
import pymongo

import pandas as pd

from heapq import nlargest

from utils import make_a_stamp, log, INFO

IP_BEANSTALK, PORT_BEANSTALK = "rongqide-Mac-mini.local", 11300
#IP_BEANSTALK = "sakaes-MacBook-Pro.local"
TIMEOUT_BEANSTALK=1800
TASK_BEANSTALK = "facebook_checkin_competition"

MONGODB_URL = "mongodb://{}:27017".format(IP_BEANSTALK)
MONGODB_BATCH_SIZE = 20000

MONGODB_INDEX = "row_id"
MONGODB_VALUE = "place_ids"
MONGODB_SCORE = "score"

MODE_SIMPLE = "simple"
MODE_WEIGHT = "weight"
MODE_VOTE = "vote"

FULL_SET = [0, 8607230]

def transform_to_submission_format(results, n_top):
    timestamp_start = time.time()
    csv_format = {}
    for test_id, rankings in results.items():
        test_id = str(test_id)

        csv_format.setdefault(test_id, [])

        for place_id, most_popular in nlargest(n_top, sorted(rankings.items()), key=lambda (k, v): v):
            csv_format[test_id].append(str(int(place_id)))

        csv_format[test_id] = " ".join(csv_format[test_id])

    timestamp_end = time.time()
    log("Cost {:8f} secends to transform the results to submission".format(timestamp_end-timestamp_start), INFO)

    return csv_format

def save_submission(filepath, results, n_top=3, is_full=[]):
    ori_count = len(results)
    if is_full and len(results) < (is_full[1]-is_full[0]):
        for test_id in range(is_full[0], is_full[1]):
            results.setdefault(str(test_id), "")

    new_count = len(results)
    if ori_count != new_count:
        log("Add {} row into the results".format(new_count-ori_count), INFO)

    for test_id, info in results.items():
        results[test_id] = " ".join(info.split(" ")[:n_top])

    pd.DataFrame(results.items(), columns=["row_id", "place_id"]).to_csv(filepath, index=False)

    log("The submission file is stored in {}".format(filepath), INFO)

def get_mongo_location(cache_workspace):
    folder_setting = os.path.basename(cache_workspace)
    folder_method = os.path.dirname(cache_workspace)
    folder_grid = os.path.dirname(folder_method)

    database = make_a_stamp("{}_{}".format(folder_grid, folder_method))
    collection = folder_setting

    log("{} {} --> {} {}".format(folder_grid, folder_method, database, collection), INFO)

    return database, collection

def _import_hdb(configuration, m):
    mongo = pymongo.MongoClient(MONGODB_URL)

    workspace, cache_workspace, submission_workspace = configuration.get_workspace(m, False)
    log("The workspace is {}".format(workspace), INFO)

    filepath_pkl = os.path.join(cache_workspace, "final_results.pkl")
    log("The filepath_pkl is {}".format(filepath_pkl), INFO)

    database, collection = get_mongo_location(cache_workspace)
    import_hdb(filepath_pkl, mongo[database][collection])

    mongo.close()

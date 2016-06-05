#!/usr/bin/env python

import os
import sys

import pandas as pd
import numpy as np

import zlib
import json
import pickle

import pymongo
import beanstalkc

from utils import log, make_a_stamp
from utils import DEBUG, INFO, WARN
from facebook_utils import IP_BEANSTALK, PORT_BEANSTALK, TIMEOUT_BEANSTALK, MONGODB_URL, MONGODB_INDEX
from facebook_strategy import StrategyEngine
from facebook_learning import KDTreeEngine, MostPopularEngine, ClassifierEngine, ProcessThread

# beanstalk client
TALK = None

# mongoDB
CLIENT = None

def init(task="facebook_checkin_competition"):
    global IP_BEANSTALK, PORT_BEANSTALK, TALK, CLIENT, CONNECTION

    TALK = beanstalkc.Connection(host=IP_BEANSTALK, port=PORT_BEANSTALK)
    TALK.watch(task)

    worker()

def get_mongo_location(cache_workspace):
    database = make_a_stamp(os.path.basename(os.path.dirname(os.path.dirname(cache_workspace))))
    collection = make_a_stamp("{}_{}".format(os.path.basename(os.path.dirname(cache_workspace)), os.path.basename(cache_workspace)))

    return database, collection

def worker():
    global CLIENT, CONNECTION, MONGODB_URL
    CLIENT = pymongo.MongoClient(MONGODB_URL)

    global TALK, TIMEOUT_BEANSTALK

    strategy, is_accuracy, is_exclude_outlier, is_testing = None, False, False, False
    strategy_engine = StrategyEngine(strategy, is_accuracy, is_exclude_outlier, is_testing)

    while True:
        job = TALK.reserve(timeout=TIMEOUT_BEANSTALK)
        if job != None:
            try:
                o = json.loads(zlib.decompress(job.body))

                method, strategy, setting = o["method"], o["strategy"], o["setting"]
                n_top, criteria = o["n_top"], o["criteria"]
                is_normalization, is_accuracy, is_exclude_outlier, is_testing = o["is_normalization"], o["is_accuracy"], o["is_exclude_outlier"], o["is_testing"]

                cache_workspace = o["cache_workspace"]
                database, collection = get_mongo_location(cache_workspace)

                mongo = CLIENT[database][collection]
                mongo.create_index(MONGODB_INDEX)

                filepath_train, filepath_test = pickle.loads(o["filepath_training"]), pickle.loads(o["filepath_testing"])

                strategy_engine.strategy = strategy
                strategy_engine.is_accuracy = is_accuracy
                strategy_engine.is_exclude_outlier = is_exclude_outlier
                strategy_engine.is_testing = is_testing

                filepath_train_pkl, f = None, None
                ave_x, std_x, ave_y, std_y = None, None, None, None

                top = None
                is_pass = True
                if method == StrategyEngine.STRATEGY_MOST_POPULAR:
                    most_popular_engine = MostPopularEngine(cache_workspace, n_top, is_testing)

                    metrics, (min_x, len_x), (min_y, len_y), (ave_x, std_x), (ave_y, std_y) =\
                        strategy_engine.get_most_popular_metrics(filepath_train, filepath_train_pkl, f, n_top, criteria[0], criteria[1], is_normalization)

                    test_id, test_x = get_testing_dataset(filepath_test, method, is_normalization, ave_x, std_x, ave_y, std_y)
                    if test_id == None or test_x == None:
                        log("Empty file in {}".format(filepath_test), WARN)
                        is_pass = False
                    else:
                        top = most_popular_engine.process(test_id, test_x, metrics, (strategy_engine.position_transformer,
                                                                                     (min_x, len_x, criteria[0]),
                                                                                     (min_y, len_y, criteria[1])),
                                                                                     is_cache=False)
                elif method == StrategyEngine.STRATEGY_KDTREE:
                    kdtree_engine = KDTreeEngine(cache_workspace, n_top, is_testing)

                    metrics, mapping, score, (ave_x, std_x), (ave_y, std_y) = strategy_engine.get_kdtree(filepath_train, filepath_train_pkl, f, n_top, is_normalization)

                    test_id, test_x = ProcessThread.get_testing_dataset(filepath_test, method, is_normalization, ave_x, std_x, ave_y, std_y)
                    if test_id == None or test_x == None:
                        log("Empty file in {}".format(filepath_test), WARN)
                        is_pass = False
                    else:
                        top = kdtree_engine.process(test_id, test_x, metrics, (mapping, score), is_cache=False)
                elif method == StrategyEngine.STRATEGY_XGBOOST:
                    classifier_engine = ClassifierEngine(cache_workspace, n_top, is_testing)
                    log("The setting of XGC is {}".format(setting), INFO)

                    metrics, (ave_x, std_x), (ave_y, std_y) = strategy_engine.get_xgboost_classifier(filepath_train, f, n_top, is_normalization, **setting)

                    test_id, test_x = ProcessThread.get_testing_dataset(filepath_test, method, is_normalization, ave_x, std_x, ave_y, std_y)
                    if not bool(test_id) or not bool(test_x):
                        log("Empty file in {}".format(filepath_test), WARN)
                        is_pass = False
                    else:
                        top = classifier_engine.process(test_id, test_x, metrics, is_cache=False)
                elif method == StrategyEngine.STRATEGY_RANDOMFOREST:
                    classifier_engine = ClassifierEngine(cache_workspace, n_top, is_testing)
                    log("The setting of RFC is {}".format(setting), INFO)

                    metrics, (ave_x, std_x), (ave_y, std_y) = strategy_enging.get_randomforest_classifier(filepath_train, f, n_top, is_normalization, **setting)

                    test_id, test_x = ProcessThread.get_testing_dataset(filepath_test, method, is_normalization, ave_x, std_x, ave_y, std_y)
                    if test_id == None or test_x == None:
                        log("Empty file in {}".format(filepath_test), WARN)
                        is_pass = False
                    else:
                        top = classifier_engine.process(test_id, test_x, metrics, is_cache=False)
                else:
                    log("illegial method - {}".format(method), WARN)
                    is_pass = False

                if is_pass:
                    count = 0
                    pool = []
                    for test_id, place_ids in top.items():
                        if place_ids:
                            r = {"row_id": test_id, "place_ids": []}

                            for place_id, score in place_ids.items():
                                r["place_ids"].append({"place_id": int(place_id), "score": score})

                            pool.append(r)
                            #count += mongo.update({"row_id": test_id}, {"$set": r}, upsert=True)["ok"]

                    mongo.insert_many(pool)
                    log("Insert {} records into the {}-{}".format(len(pool), database, collection), INFO)


                job.delete()
            except Exception as e:
                log("Error occurs, {}".format(e), WARN)

                # ('delete', 'NOT_FOUND', [])
                if str(e).find("delete") != -1 and str(e).find("NOT_FOUND") != -1:
                    pass
                else:
                    raise

if __name__ == "__main__":
    init()

    TALK.close()
    CLIENT.close()

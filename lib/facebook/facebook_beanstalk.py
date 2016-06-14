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
    folder_setting = os.path.basename(cache_workspace)
    folder_method = os.path.dirname(cache_workspace)
    folder_grid = os.path.dirname(folder_method)

    database = make_a_stamp("{}_{}".format(folder_grid, folder_method))
    collection = folder_setting

    log("{} {} --> {} {}".format(folder_grid, folder_method, database, collection), INFO)

    return database, collection

def worker():
    global CLIENT
    CLIENT = pymongo.MongoClient(MONGODB_URL)

    global TALK

    strategy, is_accuracy, is_exclude_outlier, is_testing = None, False, False, False
    strategy_engine = StrategyEngine(strategy, is_accuracy, is_exclude_outlier, is_testing)

    while True:
        job = TALK.reserve(timeout=TIMEOUT_BEANSTALK)
        if job:
            try:
                o = json.loads(zlib.decompress(job.body))

                job_id = o["id"]
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

                    test_id, test_x =  StrategyEngine.get_testing_dataset(filepath_test, method, is_normalization, ave_x, std_x, ave_y, std_y)
                    if np.any(test_id):
                        top = most_popular_engine.process(test_id, test_x, metrics, (strategy_engine.position_transformer,
                                                                                     (min_x, len_x, criteria[0]),
                                                                                     (min_y, len_y, criteria[1])),
                                                                                     is_cache=False)
                    else:
                        log("Empty file in {}".format(filepath_test), WARN)
                        is_pass = False
                elif method == StrategyEngine.STRATEGY_KDTREE:
                    kdtree_engine = KDTreeEngine(cache_workspace, n_top, is_testing)

                    metrics, mapping, score, (ave_x, std_x), (ave_y, std_y) = strategy_engine.get_kdtree(filepath_train, filepath_train_pkl, f, n_top, is_normalization)

                    test_id, test_x = StrategyEngine.get_testing_dataset(filepath_test, method, is_normalization, ave_x, std_x, ave_y, std_y)
                    if np.any(test_id):
                        top = kdtree_engine.process(test_id, test_x, metrics, (mapping, score), is_cache=False)
                    else:
                        log("Empty file in {}".format(filepath_test), WARN)
                        is_pass = False
                elif method in [StrategyEngine.STRATEGY_XGBOOST, StrategyEngine.STRATEGY_RANDOMFOREST, StrategyEngine.STRATEGY_KNN]:
                    classifier_engine = ClassifierEngine(cache_workspace, n_top, is_testing)

                    if method == StrategyEngine.STRATEGY_XGBOOST:
                        metrics, (ave_x, std_x), (ave_y, std_y) = strategy_engine.get_xgboost_classifier(filepath_train, f, n_top, is_normalization, **setting)
                        log("The setting of XGC is {}".format(setting), INFO)
                    elif method == StrategyEngine.STRATEGY_RANDOMFOREST:
                        metrics, (ave_x, std_x), (ave_y, std_y) = strategy_engine.get_randomforest_classifier(filepath_train, f, n_top, is_normalization, **setting)
                        log("The setting of RFC is {}".format(setting), INFO)
                    elif method == StrategyEngine.STRATEGY_KNN:
                        metrics, (ave_x, std_x), (ave_y, std_y) = strategy_engine.get_knn_classifier(filepath_train, f, n_top, is_normalization, **setting)
                        log("The setting of KNN is {}".format(setting), INFO)

                    test_id, test_x = StrategyEngine.get_testing_dataset(filepath_test, method, is_normalization, ave_x, std_x, ave_y, std_y)
                    if np.any(test_id):
                        top = classifier_engine.process(test_id, test_x, metrics, is_cache=False)
                    else:
                        log("Empty file in {}".format(filepath_test), WARN)
                        is_pass = False
                else:
                    log("illegial method - {}".format(method), WARN)
                    is_pass = False

                if is_pass:
                    pool = []
                    for test_id, place_ids in top.items():
                        if place_ids:
                            r = {"grid": job_id, "row_id": test_id, "place_ids": []}

                            for place_id, score in place_ids.items():
                                r["place_ids"].append({"place_id": int(place_id), "score": score})

                            pool.append(r)

                    mongo.insert_many(pool)
                    log("{}. Insert {} records into the {}-{}".format(job_id, len(pool), database, collection), INFO)

                job.delete()
            except Exception as e:
                log("Error occurs, {}".format(e), WARN)

                # ('delete', 'NOT_FOUND', [])
                if str(e).find("delete") != -1 and str(e).find("NOT_FOUND") != -1:
                    pass
                else:
                    job.delete()

if __name__ == "__main__":
    init()

    CLIENT.close()
    TALK.close()

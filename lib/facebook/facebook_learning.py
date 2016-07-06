#!/usr/bin/env python

import os
import sys
import math
import time
import glob

import json
import zlib
import pickle
import beanstalkc

import threading
import Queue

import pandas as pd
import numpy as np

from scandir import scandir
from operator import itemgetter
from scipy import stats

from facebook_utils import IP_BEANSTALK, PORT_BEANSTALK, TASK_BEANSTALK, TASK_BEANSTALK, TIMEOUT_BEANSTALK
from facebook_strategy import StrategyEngine
from utils import log, DEBUG, INFO, WARN, ERROR
from utils import create_folder, make_a_stamp
from load import save_cache, load_cache

class BaseEngine(object):
    def __init__(self, n_top, is_testing):
        self.n_top = n_top
        self.is_testing = is_testing

    def process(self, test_ids, test_xs, metrics, others):
        raise NotImplementedError

class MostPopularEngine(BaseEngine):
    def process(self, test_ids, test_xs, metrics, others):
        transformer, range_x, range_y = others

        top = {}
        count_missing = 0
        for test_id, test_x in zip(test_ids, test_xs):
            top.setdefault(test_id, {})

            key_x = test_x[0]
            if range_x[1] > 0:
                key_x = transformer(test_x[0], range_x[0], range_x[1], range_x[2])

            key_y = test_x[1]
            if range_y[1] > 0:
                key_y = transformer(test_x[1], range_y[0], range_y[1], range_y[2])

            key = "{}-{}".format(key_x, key_y)
            if key in metrics:
                for place_id, score in metrics[key]:
                    top[test_id].setdefault(place_id, 0)
                    top[test_id][place_id] += score
            else:
                log("The ({} ----> {}) of {} is not in metrics".format(test_x, key, test_id), DEBUG)

                count_missing += 1

        log("The missing ratio is {}/{}={:4f}".format(count_missing, len(top), float(count_missing)/len(top)), INFO)

        return top

class KDTreeEngine(BaseEngine):
    def process(self, test_ids, test_xs, metrics, others):
        mapping, score = others

        top = {}

        distance, ind = metrics.query(test_xs, k=min(self.n_top, len(mapping)))
        for idx, loc in enumerate(ind):
            test_id = test_ids[idx]

            top.setdefault(test_id, {})
            for loc_idx, locc in enumerate(loc):
                place_id = mapping[locc]

                top[test_id].setdefault(place_id, 0)

                d = distance[idx][loc_idx]
                if d != 0:
                    top[test_id][place_id] += -1.0*np.log(distance[idx][loc_idx])*score[locc]

        return top

class ClassifierEngine(BaseEngine):
    def process(self, test_ids, test_xs, metrics, others=None):
        top = {}

        predicted_proba = metrics.predict_proba(test_xs)

        pool = [dict(zip(metrics.classes_, probas)) for probas in predicted_proba]
        for idx, pair in enumerate(pool):
            test_id = test_ids[idx]
            top.setdefault(test_id, {})

            for place_id, proba in sorted(pair.items(), key=(lambda (k, v): v), reverse=True)[:self.n_top]:
                if proba > 0:
                    top[test_id].setdefault(place_id, 0)
                    top[test_id][place_id] += 10**proba

        return top

class BaseCalculatorThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        threading.Thread.__init__(self, group=group, target=target, name=name, verbose=verbose)

        self.args = args

        for key, value in kwargs.items():
            setattr(self, key, value)

        if not hasattr(self, "weights"):
            self.weights = 1

    def update_results(self, results):
        for test_id, clusters in results.items():
            self.results.setdefault(test_id, {})

            for place_id, score in clusters.items():
                self.results[test_id].setdefault(place_id, 0)
                self.results[test_id][place_id] += score

    def run(self):
        while True:
            timestamp_start = time.time()

            test_ids, test_xs, metrics, others = self.queue.get()

            top = self.process(test_ids, test_xs, metrics, others)
            self.update_results(top)

            self.queue.task_done()

            timestamp_end = time.time()
            log("Cost {:4f} secends to finish this batch job({} - {}, {}) getting TOP-{} clusters. The remaining size of queue is {}".format(\
                timestamp_end-timestamp_start, test_ids[0], test_ids[-1], len(top), self.n_top, self.queue.qsize()), INFO)

    def process(self, test_ids, test_xs, metrics, others):
        raise NotImplementedError

class KDTreeThread(BaseCalculatorThread):
    def process(self, test_ids, test_xs, metrics, others):
        return self.kdtree_engine.process(test_ids, test_xs, metrics, others)

def process(method, workspaces, criteria, strategy, is_accuracy, is_exclude_outlier, is_normalization, is_testing, dropout,
            n_top=3):

    global IP_BEANSTALK, PORT_BEANSTALK, TASK_BEANSTALK

    results = {}
    workspace, database, collections = workspaces

    IP_BEANSTALK, PORT_BEANSTALK = "rongqide-Mac-mini.local", 11300
    talk = beanstalkc.Connection(host=IP_BEANSTALK, port=PORT_BEANSTALK)
    talk.use(TASK_BEANSTALK)

    priority = int(time.time())
    for filepath_train in glob.iglob(workspace):
        if filepath_train.find(".csv") != -1 and filepath_train.find("test.csv") == -1 and filepath_train.find("submission") == -1:
            filepath_test = filepath_train.replace("train", "test")

            # Avoid the empty file
            if os.path.exists(filepath_test):
                # workaround
                threshold_x, threshold_y = 7.85, 8.4
                filename = os.path.basename(filepath_test).replace(".csv", "")
                x, y  = filename.split("_")
                x = float(x)
                y = float(y)
                if x < threshold_x or (x == threshold_x and y <= threshold_y):
                    continue

                df_train, df_test = None, StrategyEngine.get_dataframe(filepath_test)
                if strategy == "native":
                    df_train = StrategyEngine.get_dataframe(filepath_train, 2, dropout)
                else:
                    df_train = StrategyEngine.get_dataframe(filepath_train, 1, dropout)

                if df_train.shape[0] < 1:
                    log("Skip the filepath_train due to the empty file({})".format(filepath_train), INFO)
                    continue

                log("Ready to put {} into the queue".format(filepath_test), INFO)
                i = 0
                for collection, setting in collections.items():
                    string = {"id": os.path.basename(filepath_test),
                              "method": method,
                              "strategy": strategy,
                              "setting": setting,
                              "n_top": n_top,
                              "criteria": criteria,
                              "is_normalization": is_normalization,
                              "is_accuracy": is_accuracy,
                              "is_exclude_outlier": is_exclude_outlier,
                              "is_testing": is_testing,
                              "database": database,
                              "collection": collection,
                              "filepath_training": pickle.dumps(df_train),
                              "filepath_testing": pickle.dumps(df_test)}

                    log("{} - {} with {}".format(method, setting, priority+i), INFO)
                    talk.put(zlib.compress(json.dumps(string)), priority=priority+i, ttr=TIMEOUT_BEANSTALK)

                    i += 1

    talk.close()

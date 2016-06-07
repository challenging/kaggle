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

from facebook_utils import IP_BEANSTALK, PORT_BEANSTALK, TASK_BEANSTALK, TASK_BEANSTALK
from facebook_strategy import StrategyEngine
from utils import log, DEBUG, INFO, WARN, ERROR
from utils import create_folder, make_a_stamp
from load import save_cache, load_cache

class BaseEngine(object):
    def __init__(self, cache_workspace, n_top, is_testing):
        self.cache_workspace = cache_workspace
        self.n_top = n_top
        self.is_testing = is_testing

    def process(self, test_ids, test_xs, metrics, others, is_cache=True):
        raise NotImplementedError

    def get_filepath(self, test_ids, test_xs):
        stamp = make_a_stamp(str(test_ids) + str(test_xs))
        return os.path.join(self.cache_workspace, "{}.pkl".format(stamp))

class MostPopularEngine(BaseEngine):
    def process(self, test_ids, test_xs, metrics, others, is_cache=True):
        transformer, range_x, range_y = others

        filepath = self.get_filepath(test_ids, test_xs)

        top = {}

        if is_cache:
            top = load_cache(filepath)

        if not top or self.is_testing:
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

            if not self.is_testing and is_cache:
                save_cache(top, filepath)

        return top

class KDTreeEngine(BaseEngine):
    def process(self, test_ids, test_xs, metrics, others, is_cache=True):
        mapping, score = others

        filepath = self.get_filepath(test_ids, test_xs)

        top = {}

        if is_cache:
            top = load_cache(filepath)

        if not top or self.is_testing:
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

            if not self.is_testing and is_cache:
                save_cache(top, filepath)

        return top

class ClassifierEngine(BaseEngine):
    def process(self, test_ids, test_xs, metrics, others=None, is_cache=True):
        filepath = self.get_filepath(test_ids, test_xs)

        top = {}

        if is_cache:
            top = load_cache(filepath)

        if not top or self.is_testing:
            predicted_proba = metrics.predict_proba(test_xs)

            pool = [dict(zip(metrics.classes_, probas)) for probas in predicted_proba]
            for idx, pair in enumerate(pool):
                test_id = test_ids[idx]
                top.setdefault(test_id, {})

                for place_id, proba in sorted(pair.items(), key=(lambda (k, v): v), reverse=True)[:self.n_top]:
                    top[test_id].setdefault(place_id, 0)
                    top[test_id][place_id] += 10**proba

            if not self.is_testing and is_cache:
                save_cache(top, filepath)

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

            top = self.process(test_ids, test_xs, metrics, others, is_cache=True)
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

class ProcessThread(BaseCalculatorThread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        threading.Thread.__init__(self, group=group, target=target, name=name, verbose=verbose)

        self.args = args

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.method = self.m[0]
        self.setting = self.m[1]

        self.strategy_engine = StrategyEngine(self.strategy, self.is_accuracy, self.is_exclude_outlier, self.is_testing)
        self.kdtree_engine = KDTreeEngine(self.cache_workspace, self.n_top, self.is_testing)
        self.most_popular_engine = MostPopularEngine(self.cache_workspace, self.n_top, self.is_testing)
        self.classifier_engine = ClassifierEngine(self.cache_workspace, self.n_top, self.is_testing)

        # KDTreeThread
        self.queue_kdtree = Queue.Queue()
        for idx in range(0, self.n_jobs):
            thread = KDTreeThread(kwargs={"queue": self.queue_kdtree, "results": self.results, "kdtree_engine": self.kdtree_engine, "cache_workspace": self.cache_workspace, "is_testing": self.is_testing, "n_top": self.n_top})
            thread.setDaemon(True)
            thread.start()

    @staticmethod
    def get_testing_dataset(filepath_test, method, is_normalization, ave_x, std_x, ave_y, std_y):
        test_id, test_x = None, None

        df = StrategyEngine.get_dataframe(filepath_test)
        log("There are {} reocrds in {}".format(df.values.shape, filepath_test if isinstance(filepath_test, str) else ""), INFO)

        test_id = df["row_id"].values
        if method in [StrategyEngine.STRATEGY_XGBOOST, StrategyEngine.STRATEGY_RANDOMFOREST]:
            d_times = StrategyEngine.get_d_time(df["time"].values)

            df["hourofday"] = d_times.hour
            df["dayofmonth"] = d_times.day
            df["weekday"] = d_times.weekday
            df["monthofyear"] = d_times.month
            df["year"] = d_times.year

            if is_normalization:
                df["x"] = (df["x"] - ave_x) / (std_x + 0.00000001)
                df["y"] = (df["y"] - ave_y) / (std_y + 0.00000001)

            test_x = df[["x", "y", "accuracy", "hourofday", "dayofmonth", "monthofyear", "weekday", "year"]].values
        else:
            if is_normalization:
                df["x"] = (df["x"] - ave_x) / (std_x + 0.00000001)
                df["y"] = (df["y"] - ave_y) / (std_y + 0.00000001)

            test_x = df[["x", "y"]].values

        return test_id, test_x

    def run(self):
        while True:
            timestamp_start = time.time()

            filepath_train = self.queue.get()
            filename = os.path.basename(filepath_train)
            folder = os.path.dirname(filepath_train)

            filepath_train_pkl = os.path.join(os.path.dirname(self.cache_workspace), "train", "{}.{}.pkl".format(self.strategy, make_a_stamp(filepath_train)))
            create_folder(filepath_train_pkl)

            filepath_test = filepath_train.replace("train", "test")

            metrics, mapping = None, None
            ave_x, std_x, ave_y, std_y = None, None, None, None

            if self.method == self.strategy_engine.STRATEGY_MOST_POPULAR:
                f = os.path.join(self.cache_workspace, "{}.{}.pkl".format(self.strategy_engine.get_most_popular_metrics.__name__.lower(), filename))
                metrics, (min_x, len_x), (min_y, len_y), (ave_x, std_x), (ave_y, std_y) =\
                    self.strategy_engine.get_most_popular_metrics(filepath_train, filepath_train_pkl, f, self.n_top, self.criteria[0], self.criteria[1], self.is_normalization)
            elif self.method == self.strategy_engine.STRATEGY_KDTREE:
                f = os.path.join(self.cache_workspace, "{}.{}.pkl".format(self.strategy_engine.get_kdtree.__name__.lower(), filename))
                metrics, mapping, score, (ave_x, std_x), (ave_y, std_y) = self.strategy_engine.get_kdtree(filepath_train, filepath_train_pkl, f, self.n_top, self.is_normalization)
            elif self.method == self.strategy_engine.STRATEGY_XGBOOST:
                f = os.path.join(self.cache_workspace, "{}.{}.pkl".format(self.strategy_engine.get_xgboost_classifier.__name__.lower(), filename))
                log("The setting of XGC is {}".format(self.setting), INFO)
                metrics, (ave_x, std_x), (ave_y, std_y) = self.strategy_engine.get_xgboost_classifier(filepath_train, f, self.n_top, self.is_normalization, **self.setting)
            elif self.method == self.strategy_engine.STRATEGY_RANDOM_FOREST:
                f = os.path.join(self.cache_workspace, "{}.{}.pkl".format(self.strategy_engine.get_randomforest_classifier.__name__.lower(), filename))
                log("The setting of RFC is {}".format(self.setting), INFO)
                metrics, (ave_x, std_x), (ave_y, std_y) = self.strategy_enging.get_randomforest_classifier(filepath_train, f, self.n_top, self.is_normalization, **self.setting)
            else:
                log("Not implement this method, {}".format(self.method), ERROR)
                raise NotImplementedError

            if os.path.exists(filepath_test):
                test_id, test_x = get_testing_dataset(filepath_test, self.method, self.is_normalization, ave_x, std_x, ave_y, std_y)

                top = []
                if self.method == self.strategy_engine.STRATEGY_KDTREE:
                    #top = self.kdtree_engine.process(test_id, test_x, metrics, (mapping, score))
                    for idx in range(0, test_id.shape[0]/self.batch_size+1):
                        idx_start, idx_end = idx*self.batch_size, min(test_id.shape[0], (idx+1)*self.batch_size)
                        size = test_id[idx_start:idx_end].shape[0]

                        if size > 0:
                            self.queue_kdtree.put((test_id[idx_start:idx_end], test_x[idx_start:idx_end], metrics, (mapping, score)))

                    log("There are {} items in the Queue of KDTree".format(self.queue_kdtree.qsize()), INFO)
                    self.queue_kdtree.join()
                elif self.method == self.strategy_engine.STRATEGY_MOST_POPULAR:
                    top = self.most_popular_engine.process(test_id, test_x, metrics, (self.strategy_engine.position_transformer,
                                                                                      (min_x, len_x, self.criteria[0]),
                                                                                      (min_y, len_y, self.criteria[1])))
                    self.update_results(top)
                elif self.method in [self.strategy_engine.STRATEGY_XGBOOST, self.strategy_engine.STRATEGY_RANDOMFOREST]:
                    top = self.classifier_engine.process(test_id, test_x, metrics)
                    self.update_results(top)
                else:
                    raise NotImplementedError
            else:
                log("Not Found the testing file in {}".format(filepath_test), WARN)

            self.queue.task_done()

            timestamp_end = time.time()
            log("Cost {:8f} seconds to finish the prediction of {} by {}, {}".format(timestamp_end-timestamp_start, filepath_train, self.method, self.queue.qsize()), INFO)

def process(m, workspaces, filepath_pkl, batch_size, criteria, strategy, is_accuracy, is_exclude_outlier, is_normalization, is_beanstalk, is_testing,
            n_top=3, n_jobs=8):

    results = {}
    method, setting = m
    workspace, cache_workspace, output_workspace = workspaces

    if is_beanstalk:
        global IP_BEANSTALK, PORT_BEANSTALK, TASK_BEANSTALK

        talk = beanstalkc.Connection(host=IP_BEANSTALK, port=PORT_BEANSTALK)
        talk.use(TASK_BEANSTALK)

        for filepath_train in glob.iglob(workspace):
            if filepath_train.find(".csv") != -1 and filepath_train.find("test.csv") == -1 and filepath_train.find("submission") == -1:
                filepath_test = filepath_train.replace("train", "test")

                # Avoid the empty file
                if os.stat(filepath_train).st_size > 34 and os.path.exists(filepath_test):
                    df_train = None
                    if strategy == "native":
                        df_train = StrategyEngine.get_dataframe(filepath_train, 2)
                    else:
                        df_train = StrategyEngine.get_dataframe(filepath_train, 1)

                    df_test = StrategyEngine.get_dataframe(filepath_test)

                    string = {"method": method,
                              "strategy": strategy,
                              "setting": setting,
                              "n_top": n_top,
                              "criteria": criteria,
                              "is_normalization": is_normalization,
                              "is_accuracy": is_accuracy,
                              "is_exclude_outlier": is_exclude_outlier,
                              "is_testing": is_testing,
                              "cache_workspace": cache_workspace,
                              "filepath_training": pickle.dumps(df_train),
                              "filepath_testing": pickle.dumps(df_test)}

                    request = zlib.compress(json.dumps(string))

                    talk.put(request, ttr=600)

        talk.close()

        return None
    else:
        for folder in [os.path.join(cache_workspace, "1.txt"), os.path.join(output_workspace, "1.txt")]:
            create_folder(folder)

        results = load_cache(filepath_pkl, is_hdb=True, simple_mode=True)
        if not results:
            queue = Queue.Queue()
            for filepath_train in glob.iglob(workspace):
                if filepath_train.find(".csv") != -1 and filepath_train.find("test.csv") == -1 and filepath_train.find("submission") == -1:
                    # Avoid the empty file
                    if os.stat(filepath_train).st_size > 34:
                        queue.put(filepath_train)
                        log("Push {} in queue".format(filepath_train), INFO)

            if queue.qsize() == 0:
                log("Not found any files in {}".format(workspace), WARN)
                return None

            log("For {}({}), there are {} files in queue".format(method, criteria, queue.qsize()), INFO)

            for idx in range(0, n_jobs):
                thread = ProcessThread(kwargs={"queue": queue,
                                               "results": results,
                                               "m": m,
                                               "criteria": criteria,
                                               "strategy": strategy,
                                               "batch_size": batch_size,
                                               "cache_workspace": cache_workspace,
                                               "submission_workspace": output_workspace,
                                               "is_accuracy": is_accuracy,
                                               "is_exclude_outlier": is_exclude_outlier,
                                               "is_normalization": is_normalization,
                                               "is_testing": is_testing,
                                               "n_top": n_top,
                                               "n_jobs": n_jobs})
                thread.setDaemon(True)
                thread.start()
            queue.join()

            log("Start to save results in cache", INFO)
            save_cache(results, filepath_pkl, is_hdb=True)
            log("Finish saving cache", INFO)

        log("There are {} records in results".format(len(results)), INFO)

        return resutls

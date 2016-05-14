#!/usr/bin/env python

import os
import sys
import math
import glob
import time
import datetime

import threading
import Queue

import pandas as pd
import numpy as np

from scandir import scandir
from heapq import nlargest
from scipy import stats
from scipy.spatial.distance import euclidean

from sklearn.neighbors import NearestCentroid, DistanceMetric

from utils import log, DEBUG, INFO, WARN
from utils import create_folder
from load import save_cache, load_cache

class BaseCalculatorThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        threading.Thread.__init__(self, group=group, target=target, name=name, verbose=verbose)

        self.args = args

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.results = {}

    def update_results(self, results):
        for test_id, clusters in results.items():
            self.results.setdefault(test_id, {})

            for adjust, place_id in enumerate(clusters):
                self.results[test_id].setdefault(place_id, 0)
                self.results[test_id][place_id] += (len(clusters) - adjust)*1.0

class MostPopularThread(BaseCalculatorThread):
    def run(self):
        while True:
            timestamp_start = time.time()
            test_ids, test_xs = self.queue.get()

            top = {}
            for test_id, test_x in zip(test_ids, test_xs):
                top.setdefault(test_id, [])

                key = self.transformer(test_x[0], test_x[1])
                if key in self.metrics:
                    for place_id, most_popular in nlargest(self.n_top, sorted(self.metrics[key].items()), key=lambda (k, v): v):
                        top[test_id].append(place_id)
                else:
                    log("The ({} ----> {}) of {} is not in metrics".format(test_x, key, test_id), WARN)

            self.update_results(top)

            self.queue.task_done()

            timestamp_end = time.time()
            log("Cost {:4f} secends to finish this batch job({} - {}, {}) getting TOP-{} clusters".format(timestamp_end-timestamp_start, test_ids[0], test_ids[-1], len(self.results), self.n_top), INFO)

class KDTreeThread(threading.Thread):
    def run(self):
        pass

class CalculateDistanceThread(BaseCalculatorThread):
    def run(self):
        while True:
            timestamp_start = time.time()
            test_ids, test_xs = self.queue.get()

            rankings = {}
            for test_id, test_x in zip(test_ids, test_xs):
                rankings.setdefault(test_id, {})
                for metrics, place_id in self.metrics.items():
                    rankings[test_id].setdefault(place_id, self.distance(test_x, list(metrics)))

            top = {}
            for test_id, ranking in rankings.items():
                for place_id, distance in sorted(ranking.items(), key=lambda (k, v): v)[:self.n_top]:
                    top[test_id].append(str(place_id))

            self.update_results(top)

            self.queue.task_done()

            timestamp_end = time.time()
            log("Cost {:4f} secends to finish this batch job({} - {}) getting TOP-{} clusters".format(timestamp_end-timestamp_start, test_ids[0], test_ids[-1], self.n_top), DEBUG)

class StrategyEngine(object):
    STRATEGY_DISTANCE = "distance"
    STRATEGY_MOST_POPULAR = "most_popular"

    def __init__(self, is_accuracy, is_exclude_outlier, is_testing):
        self.is_accuracy = is_accuracy
        self.is_exclude_outlier = is_exclude_outlier
        self.is_testing = is_testing

    def data_preprocess(self, df):
        df_target = df
        if self.is_exclude_outlier:
              df_target = df_target[(stats.zscore(df_target["x"]) < 3) & (stats.zscore(df_target["y"]) < 3) & (stats.zscore(df_target["accuracy"]) < 3)]

        return df_target

    def get_nearest_centroid(self, filepath):
        df = pd.read_csv(filepath)

        for place_id in df["place_id"].unique():
            df_target = df[df["place_id"] == place_id]
            df_target = self.data_preprocess(df_target[(stats.zscore(df_target["x"]) < 3) & (stats.zscore(df_target["y"]) < 3) & (stats.zscore(df_target["accuracy"]) < 3)])

            x, y, accuracy = df_target["x"].mean(), df_target["y"].mean(), df_target["accuracy"].mean()

            yield place_id, (x, y, accuracy)

    def get_centroid_distance_metrics(self, filepath, filepath_pkl):
        metrics = {}

        timestamp_start = time.time()
        if os.path.exists(filepath_pkl):
            metrics = load_cache(filepath_pkl)
        else:
            for place_id, m in self.get_nearest_centroid(filepath):
                metrics[m if self.is_accuracy else m[:2]] = place_id
        timestamp_end = time.time()
        log("Cost {:4} seconds to build up the centroid metrics".format(timestamp_end-timestamp_start), INFO)

        save_cache(metrics, filepath_pkl)

        return metrics

    def most_popular_transformer(self, x, y):
        range = 800
        ix = math.floor(range*x/10)
        if ix < 0:
            ix = 0
        if ix >= range:
            ix = range-1

        iy = math.floor(range*y/10)
        if iy < 0:
            iy = 0
        if iy >= range:
            iy = range-1

        return int(ix), int(iy)

    def get_most_popular_metrics(self, filepath, filepath_pkl):
        metrics = {}

        timestamp_start = time.time()
        if os.path.exists(filepath_pkl):
            metrics = load_cache(filepath_pkl)
        else:
            with open(filepath, "rb") as INPUT:
                for line in INPUT:
                    arr = line.strip().split(",")

                    row_id = arr[0]
                    if not row_id.isdigit():
                        continue

                    x = float(arr[1])
                    y = float(arr[2])
                    accuracy = arr[3]
                    place_id = arr[5]

                    key = self.most_popular_transformer(x, y)
                    metrics.setdefault(key, {})
                    metrics[key].setdefault(place_id, 0)
                    metrics[key][place_id] += 1

            save_cache(metrics, filepath_pkl)
        timestamp_end = time.time()

        log("Cost {:8f} secends to build up the most popular solution".format(timestamp_end-timestamp_start), INFO)

        return metrics

class ProcessThread(BaseCalculatorThread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        threading.Thread.__init__(self, group=group, target=target, name=name, verbose=verbose)

        self.args = args

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.strategy_engine = StrategyEngine(self.is_accuracy, self.is_exclude_outlier, self.is_testing)
        self.results = {}

    def run(self):
        while True:
            timestamp_start = time.time()

            filepath_train = self.queue.get()
            filename = os.path.basename(filepath_train)
            folder = os.path.dirname(filepath_train)

            filepath_test = filepath_train.replace("train", "test")
            filepath_pkl = os.path.join(self.cache_workspace, "{}.{}.pkl".format(type(self).__name__.lower(), filename))
            filepath_submission = os.path.join(self.submission_workspace, "{}.{}.pkl".format(type(self).__name__.lower(), filename))

            log(filepath_pkl)

            for f in [filepath_pkl, filepath_submission]:
                create_folder(f)

            results = {}
            if not self.is_testing and os.path.exists(filepath_pkl):
                results = load_cache(filepath_pkl)
            else:
                metrics = None
                if self.method == self.strategy_engine.STRATEGY_DISTANCE:
                    f = os.path.join(self.cache_workspace, "{}.{}.pkl".format(self.strategy_engine.get_centroid_distance_metrics.__name__.lower(), filename))
                    metrics = self.strategy_engine.get_centroid_distance_metrics(filepath_train, f)
                elif self.method == self.strategy_engine.STRATEGY_MOST_POPULAR:
                    f = os.path.join(self.cache_workspace, "{}.{}.pkl".format(self.strategy_engine.get_most_popular_metrics.__name__.lower(), filename))
                    metrics = self.strategy_engine.get_most_popular_metrics(filepath_train, f)

                df = pd.read_csv(filepath_test)
                if self.is_testing:
                    df = df.head(100)
                log("There are {} reocrds in {}".format(df.values.shape, filepath_test), INFO)

                fields = ["x", "y"]
                if self.is_accuracy:
                    fields.append("accuracy")

                test_id = df["row_id"].values
                test_x = df[fields].values

                queue = Queue.Queue()
                for idx in range(0, test_id.shape[0]/self.batch_size+1):
                    idx_start, idx_end = idx*self.batch_size, min(test_id.shape[0], (idx+1)*self.batch_size)
                    queue.put((test_id[idx_start:idx_end], test_x[idx_start:idx_end]))

                threads = []
                for idx in range(0, max(1, self.n_jobs/2)):
                    thread = None
                    if self.method == self.strategy_engine.STRATEGY_DISTANCE:
                        thread = CalculateDistanceThread(kwargs={"queue": queue,
                                                                 "cache_workspace": self.cache_workspace,
                                                                 "metrics": metrics,
                                                                 "distance": euclidean,
                                                                 "n_top": self.n_top})
                    elif self.method == self.strategy_engine.STRATEGY_MOST_POPULAR:
                        thread = MostPopularThread(kwargs={"queue": queue,
                                                           "cache_workspace": self.cache_workspace,
                                                           "metrics": metrics,
                                                           "transformer": self.strategy_engine.most_popular_transformer,
                                                           "n_top": self.n_top})

                    thread.setDaemon(True)
                    thread.start()

                    threads.append(thread)

                queue.join()

                for thread in threads:
                    results.update(thread.results)

            if not self.is_testing:
                save_submission(filepath_submission, results)
                save_cache(results, filepath_pkl)

            self.update_results(results)

            self.queue.task_done()

            timestamp_end = time.time()
            log("Cost {:8f} to finish the prediction of {}".format(timestamp_end-timestamp_start, filepath_train), INFO)

def process(method, workspaces, batch_size, is_accuracy, is_exclude_outlier, is_testing, n_top=3, n_jobs=8):
    results = {}

    workspace, cache_workspace, output_workspace = workspaces
    for folder in [os.path.join(cache_workspace, "1.txt"), os.path.join(output_workspace, "1.txt")]:
        create_folder(folder)

    queue = Queue.Queue()
    for filename in scandir(os.path.join(workspace)):
        filepath_train = filename.path

        if filepath_train.find(".csv") != -1 and filepath_train.find("test.csv") == -1 and filepath_train.find("submission") == -1:
            queue.put(filepath_train)
            log("Push {} in queue".format(filepath_train), INFO)

    log("There are {} files in queue".format(queue.qsize()), INFO)

    threads = []
    for idx in range(0, n_jobs):
        thread = ProcessThread(kwargs={"queue": queue,
                                       "method": method,
                                       "batch_size": batch_size,
                                       "cache_workspace": cache_workspace,
                                       "submission_workspace": output_workspace,
                                       "is_accuracy": is_accuracy,
                                       "is_exclude_outlier": is_exclude_outlier,
                                       "is_testing": is_testing,
                                       "n_top": n_top,
                                       "n_jobs": n_jobs})
        thread.setDaemon(True)
        thread.start()

        threads.append(thread)
    queue.join()

    for thread in threads:
        for test_id, clusters in thread.results.items():
            results.setdefault(test_id, {})

            for adjust, place_id in enumerate(clusters):
                results[test_id].setdefault(place_id, 0)
                results[test_id][place_id] += (len(clusters) - adjust)*1.0

    csv_format = {}
    for test_id, rankings in results.items():
        csv_format.setdefault(test_id, [])

        for place_id, most_popular in nlargest(3, sorted(rankings.items()), key=lambda (k, v): v):
            csv_format[test_id].append(place_id.strip())

        csv_format[test_id] = " ".join(csv_format[test_id])

    return csv_format

def save_submission(filepath, results):
    pd.DataFrame(results.items(), columns=["row_id", "place_id"]).to_csv(filepath, index=False, compression="gzip")

    log("The submission file is stored in {}".format(filepath), INFO)

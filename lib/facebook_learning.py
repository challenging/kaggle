#!/usr/bin/env python

import os
import sys
import math
import glob
import copy
import time
import datetime

import pprint

import threading
import Queue

import pandas as pd
import numpy as np

from scandir import scandir
from joblib import Parallel, delayed
from operator import itemgetter
from heapq import nlargest
from scipy import stats
from scipy.spatial.distance import euclidean
from sklearn.neighbors import NearestCentroid, DistanceMetric, KDTree

from utils import log, DEBUG, INFO, WARN, ERROR
from utils import create_folder, make_a_stamp
from load import save_cache, load_cache

class BaseCalculatorThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        threading.Thread.__init__(self, group=group, target=target, name=name, verbose=verbose)

        self.args = args

        for key, value in kwargs.items():
            setattr(self, key, value)

        if not hasattr(self, "weights"):
            self.weights = 1

    def update_results(self, results, is_adjust=False):
        for test_id, clusters in results.items():
            self.results.setdefault(test_id, {})

            for adjust, place_id in enumerate(clusters):
                self.results[test_id].setdefault(place_id, 0)

                score = 1
                if is_adjust:
                    score = (len(clusters) - adjust)*1.0
                else:
                    if isinstance(clusters, dict):
                        score = clusters[place_id]

                self.results[test_id][place_id] += score*self.weights
                if test_id == 0:
                    log("{} >>>> {}".format(place_id, self.results[test_id][place_id]))

    def run(self):
        while True:
            timestamp_start = time.time()

            test_ids, test_xs = self.queue.get()

            top = self.process(test_ids, test_xs)
            self.update_results(top, True)

            self.queue.task_done()

            timestamp_end = time.time()
            log("Cost {:4f} secends to finish this batch job({} - {}, {}) getting TOP-{} clusters. The remaining size of queue is {}".format(timestamp_end-timestamp_start, test_ids[0], test_ids[-1], len(top), self.n_top, self.queue.qsize()), DEBUG)

    def process(self, test_ids, test_xs):
        raise NotImplementError

class MostPopularThread(BaseCalculatorThread):
    def process(self, test_ids, test_xs):
        stamp = make_a_stamp(str(test_ids) + str(test_xs))
        filepath = os.path.join(self.cache_workspace, "{}.pkl".format(stamp))

        top = load_cache(filepath)
        if not top or self.is_testing:
            top = {}

            for test_id, test_x in zip(test_ids, test_xs):
                top.setdefault(test_id, [])

                key = (self.transformer(test_x[0], self.range_x[0], self.range_x[1], self.range_x[2]), self.transformer(test_x[1], self.range_y[0], self.range_y[1], self.range_y[2]))
                if key in self.metrics:
                    for place_id, most_popular in self.metrics[key]:
                        top[test_id].append(place_id)

                    if test_id == 0:
                        log("{} !!!! {}".format(place_id, top[test_id][-1]))
                else:
                    log("The ({} ----> {}) of {} is not in metrics".format(test_x, key, test_id), WARN)

            if not self.is_testing:
                save_cache(top, filepath)

        return top

class KDTreeThread(BaseCalculatorThread):
    def process(self, test_ids, test_xs):
        stamp = make_a_stamp(str(test_ids) + str(test_xs))
        filepath = os.path.join(self.cache_workspace, "{}.pkl".format(stamp))

        top = load_cache(filepath)
        if not top or self.is_testing:
            top = {}

            distance, ind = self.metrics.query(test_xs, k=self.n_top)
            for idx, loc in enumerate(ind):
                test_id = test_ids[idx]

                top.setdefault(test_id, {})
                for loc_idx, locc in enumerate(loc):
                    place_id = self.mapping[locc]

                    top[test_id].setdefault(place_id, 0)
                    top[test_id][place_id] += -1.0*np.log(distance[idx][loc_idx])

            if not self.is_testing:
                save_cache(top, filepath)

        return top

class StrategyEngine(object):
    STRATEGY_DISTANCE = "distance"
    STRATEGY_MOST_POPULAR = "most_popular"
    STRATEGY_KDTREE = "kdtree"

    STRATEGY_QUEUE = Queue.Queue()

    def __init__(self, is_accuracy, is_exclude_outlier, is_testing):
        self.is_accuracy = is_accuracy
        self.is_exclude_outlier = is_exclude_outlier
        self.is_testing = is_testing

    @staticmethod
    def data_preprocess(df, results, is_accuracy, is_exclude_outlier, threshold=3):
        while True:
            timestamp_start = time.time()
            place_id = StrategyEngine.STRATEGY_QUEUE.get()

            df_target = df[df["place_id"] == place_id]
            ori_shape = df_target.shape

            df_target = df_target[(stats.zscore(df_target["x"]) < threshold) & (stats.zscore(df_target["y"]) < threshold)] if is_exclude_outlier else df_target
            new_shape = df_target.shape

            if ori_shape != new_shape:
                log("Cut off outlier to make shape from {} to {} for {}".format(ori_shape, new_shape, place_id), INFO)

            x, y = df_target["x"].mean(), df_target["y"].mean()
            accuracy = df_target["accuracy"].mean() if is_accuracy else -1

            results.append((place_id, (x, y, accuracy)))

            StrategyEngine.STRATEGY_QUEUE.task_done()

            timestamp_end = time.time()
            log("Cost {:8f} seconds to get the centroid({}, {}, {}) from {}({}). Then, the remaining size of queue is {}".format(timestamp_end-timestamp_start, x, y, accuracy, df_target.shape, ori_shape, StrategyEngine.STRATEGY_QUEUE.qsize()), DEBUG)

    @staticmethod
    def position_transformer(x, min_x, len_x, range_x="800"):
        return int(math.floor((x-min_x)/len_x*int(range_x)/10))

    def get_centroid(self, filepaath, n_jobs=6):
        df = pd.read_csv(filepath)

        results = []

        if self.is_exclude_outlier:
            timestamp_start = time.time()

            for place_id in df["place_id"].unique():
                StrategyEngine.STRATEGY_QUEUE.put(place_id)

            for idx in range(0, n_jobs):
                thread = threading.Thread(target=StrategyEngine.data_preprocess, kwargs={"df": df, "results": results, "is_accuracy": self.is_accuracy, "is_exclude_outlier": self.is_exclude_outlier})
                thread.setDaemon(thread)
                thread.start()

            StrategyEngine.STRATEGY_QUEUE.join()

            results = np.array(results)

            timestamp_end = time.time()
            log("Cost {:8f} secends to filter out the outliner".format(timestamp_end-timestamp_start), INFO)
        else:
            df["place_id"] = df["place_id"].astype(str)
            results = df[["place_id", "x", "y", "accuracy"]].values

        return results

    def get_training_dataset(self, filepath, filepath_pkl, n_top):
        info = load_cache(filepath_pkl)
        if not info or self.is_testing:
            results = self.get_centroid(filepath)
            training_dataset, mapping = results[:,1:], results[:,0]

            if not self.is_testing:
                save_cache((training_dataset, mapping), filepath_pkl)
        else:
            training_dataset, mapping = info

        return np.array(training_dataset), mapping

    def get_kdtree(self, filepath, filepath_train_pkl, filepath_pkl, n_top):
        timestamp_start = time.time()

        info = load_cache(filepath_pkl)
        if not info or self.is_testing:
            training_dataset, mapping = self.get_training_dataset(filepath, filepath_train_pkl, n_top)
            tree = KDTree(training_dataset, n_top)

            if not self.is_testing:
                save_cache((tree, mapping), filepath_pkl)
        else:
            tree, mapping = info

        timestamp_end = time.time()
        log("Cost {:8f} secends to build up the KDTree solution".format(timestamp_end-timestamp_start), INFO)

        return tree, mapping

    def get_most_popular_metrics(self, filepath, filepath_train_pkl, filepath_pkl, n_top=3, range_x=800, range_y=800):
        timestamp_start = time.time()

        info = load_cache(filepath_pkl)
        if not info or self.is_testing:
            training_dataset, mapping = self.get_training_dataset(filepath, filepath_train_pkl, n_top)

            metrics = {}

            min_x, max_x = training_dataset[:,0].min(), training_dataset[:,0].max()
            len_x = max_x - min_x

            min_y, max_y = training_dataset[:,1].min(), training_dataset[:,1].max()
            len_y = max_y - min_y

            for idx in range(0, training_dataset.shape[0]):
                x = StrategyEngine.position_transformer(training_dataset[idx,0], min_x, len_x, range_x)
                y = StrategyEngine.position_transformer(training_dataset[idx,1], min_y, len_y, range_y)
                place_id = mapping[idx]

                key = (x, y)
                metrics.setdefault(key, {})
                metrics[key].setdefault(place_id, 0)

            for key in metrics.keys():
                metrics[key] = nlargest(n_top, sorted(metrics[key].items()), key=itemgetter(1))

            log("The compression rate is {}/{}={:4f}".format(len(metrics), training_dataset.shape[0], float(len(metrics))/training_dataset.shape[0]), INFO)

            if not self.is_testing:
                save_cache((metrics, (min_x, len_x), (min_y, len_y)), filepath_pkl)
        else:
            metrics, (min_x, len_x), (min_y, len_y) = info

        timestamp_end = time.time()
        log("Cost {:8f} secends to build up the most popular solution".format(timestamp_end-timestamp_start), INFO)

        return metrics, (min_x, len_x), (min_y, len_y)

class ProcessThread(BaseCalculatorThread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        threading.Thread.__init__(self, group=group, target=target, name=name, verbose=verbose)

        self.args = args

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.strategy_engine = StrategyEngine(self.is_accuracy, self.is_exclude_outlier, self.is_testing)

    def run(self):
        while True:
            timestamp_start = time.time()

            filepath_train = self.queue.get()
            filename = os.path.basename(filepath_train)
            folder = os.path.dirname(filepath_train)

            filepath_train_pkl = os.path.join(self.cache_workspace, "train", "{}.pkl".format(make_a_stamp(filepath_train)))
            create_folder(filepath_train_pkl)

            filepath_test = filepath_train.replace("train", "test")
            filepath_submission = os.path.join(self.submission_workspace, "{}.{}.pkl".format(type(self).__name__.lower(), filename))

            metrics, mapping = None, None
            if self.method == self.strategy_engine.STRATEGY_MOST_POPULAR:
                f = os.path.join(self.cache_workspace, "{}.{}.pkl".format(self.strategy_engine.get_most_popular_metrics.__name__.lower(), filename))
                metrics, (min_x, len_x), (min_y, len_y) = self.strategy_engine.get_most_popular_metrics(filepath_train, filepath_train_pkl, f, self.n_top, self.criteria[0], self.criteria[1])
            elif self.method == self.strategy_engine.STRATEGY_KDTREE:
                f = os.path.join(self.cache_workspace, "{}.{}.pkl".format(self.strategy_engine.get_kdtree.__name__.lower(), filename))
                metrics, mapping = self.strategy_engine.get_kdtree(filepath_train, filepath_train_pkl, f, self.n_top)
            else:
                log("Not implement this method, {}".format(self.method), ERROR)
                raise NotImplementError

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

            n_threads = max(1, self.n_jobs/2)
            for idx in range(0, n_threads):
                thread = None
                if self.method == self.strategy_engine.STRATEGY_MOST_POPULAR:
                    thread = MostPopularThread(kwargs={"queue": queue,
                                                       "results": self.results,
                                                       "cache_workspace": self.cache_workspace,
                                                       "metrics": metrics,
                                                       "transformer": self.strategy_engine.position_transformer,
                                                       "range_x": (min_x, len_x, self.criteria[0]),
                                                       "range_y": (min_y, len_y, self.criteria[1]),
                                                       "is_testing": self.is_testing,
                                                       "n_top": self.n_top})
                elif self.method == self.strategy_engine.STRATEGY_KDTREE:
                    thread = KDTreeThread(kwargs={"queue": queue,
                                                  "results": self.results,
                                                  "cache_workspace": self.cache_workspace,
                                                  "metrics": metrics,
                                                  "mapping": mapping,
                                                  "is_testing": self.is_testing,
                                                  "n_top": self.n_top})
                else:
                    log("Not implement this method, {}".format(self.method), ERROR)
                    raise NotImplementError

                thread.setDaemon(True)
                thread.start()

            queue.join()

            self.queue.task_done()

            timestamp_end = time.time()
            log("Cost {:8f} seconds to finish the prediction of {} by {}".format(timestamp_end-timestamp_start, filepath_train, self.method), INFO)

def process(method, workspaces, filepath_pkl, batch_size, criteria, is_accuracy, is_exclude_outlier, is_testing, n_top=3, n_jobs=8):
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

    threads, results = [], load_cache(filepath_pkl)
    if not results:
        results = {}

        for idx in range(0, n_jobs):
            thread = ProcessThread(kwargs={"queue": queue,
                                           "results": results,
                                           "method": method,
                                           "criteria": criteria,
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

        save_cache(results, filepath_pkl)

    timestamp_start = time.time()
    csv_format = {}
    for test_id, rankings in results.items():
        csv_format.setdefault(test_id, [])

        for place_id, most_popular in nlargest(n_top, sorted(rankings.items()), key=lambda (k, v): v):
            csv_format[test_id].append(place_id)

            if test_id == 0:
                log(csv_format[test_id][-1])

        csv_format[test_id] = " ".join(csv_format[test_id])

    timestamp_end = time.time()
    log("Cost {:8f} secends to transform the results to submission".format(timestamp_end-timestamp_start), INFO)

    return csv_format

def save_submission(filepath, results, n_top=3):
    for test_id, info in results.items():
        results[test_id] = " ".join(info.split(" ")[:n_top])

    pd.DataFrame(results.items(), columns=["row_id", "place_id"]).to_csv(filepath, index=False, compression="gzip")

    log("The submission file is stored in {}".format(filepath), INFO)

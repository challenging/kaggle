#!/usr/bin/env python

import os
import sys
import math
import time

import threading
import Queue

import pandas as pd
import numpy as np

from heapq import nlargest
from scipy import stats
from sklearn.neighbors import NearestCentroid, DistanceMetric, KDTree

from utils import log, DEBUG, INFO, WARN, ERROR
from load import save_cache, load_cache

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

            if is_exclude_outlier and df_target.shape[0] > 2:
                df_target = df_target[(stats.zscore(df_target["x"]) < threshold) & (stats.zscore(df_target["y"]) < threshold)]

            new_shape = df_target.shape

            x, y = df_target["x"].mean(), df_target["y"].mean()
            accuracy = df_target["accuracy"].mean() if is_accuracy else -1

            results.append([place_id, x, y, accuracy])

            StrategyEngine.STRATEGY_QUEUE.task_done()

            timestamp_end = time.time()
            log("Cost {:8f} seconds to get the centroid({}, {}, {}) from [{} ---> {}]. Then, the remaining size of queue is {}".format(timestamp_end-timestamp_start, x, y, accuracy, ori_shape, new_shape, StrategyEngine.STRATEGY_QUEUE.qsize()), DEBUG)

    @staticmethod
    def position_transformer(x, min_x, len_x, range_x="800"):
        return int(float(x-min_x)/len_x*int(range_x))

    def get_centroid(self, filepath, n_jobs=6):
        df = pd.read_csv(filepath, dtype={"row_id": str, "place_id": str, "x":np.float32, "y":np.float32, "accuracy": np.int32, "time": np.int32})

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

        training_dataset = training_dataset.astype(np.float32)

        return training_dataset, mapping

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
                try:
                    x = StrategyEngine.position_transformer(training_dataset[idx,0], min_x, len_x, range_x)
                    y = StrategyEngine.position_transformer(training_dataset[idx,1], min_y, len_y, range_y)
                    place_id = mapping[idx]

                    key = (x, y)
                    metrics.setdefault(key, {})
                    metrics[key].setdefault(place_id, 0)
                    metrics[key][place_id] += 1
                except ValueError as e:
                    log(e)
                    log("The inforamtion of x are {},{},{}".format(training_dataset[idx,0], min_x, len_x), WARN)
                    log("The inforamtion of y are {},{},{}".format(training_dataset[idx,1], min_y, len_y), WARN)

            for key in metrics.keys():
                metrics[key] = nlargest(n_top, sorted(metrics[key].items()), key=lambda (k, v): v)

            log("The compression rate is {}/{}={:4f}".format(len(metrics), training_dataset.shape[0], float(len(metrics))/training_dataset.shape[0]), INFO)

            if not self.is_testing:
                save_cache((metrics, (min_x, len_x), (min_y, len_y)), filepath_pkl)
        else:
            metrics, (min_x, len_x), (min_y, len_y) = info

        timestamp_end = time.time()
        log("Cost {:8f} secends to build up the most popular solution".format(timestamp_end-timestamp_start), INFO)

        return metrics, (min_x, len_x), (min_y, len_y)

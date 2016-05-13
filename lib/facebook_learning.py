#!/usr/bin/env python

import os
import sys
import glob
import time
import datetime

import threading
import Queue

import pandas as pd
import numpy as np

from scipy import stats
from scipy.spatial.distance import euclidean

from sklearn.neighbors import NearestCentroid, DistanceMetric

from utils import log, INFO, create_folder
from load import save_cache, load_cache

class CalculateDistanceThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        threading.Thread.__init__(self, group=group, target=target, name=name, verbose=verbose)

        self.args = args

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.results = {}

    def run(self):
        while True:
            timestamp_start = time.time()
            test_id, test_x = self.queue.get()

            ranking = {}
            for metrics, place_id in self.centroid_metrics.items():
                ranking[place_id] = self.distance(test_x, list(metrics))

            top = {test_id: []}
            for place_id, distance in sorted(ranking.items(), key=lambda (k, v): v)[:self.n_top]:
                top[test_id].append(str(place_id))
            top[test_id] = " ".join(top[test_id])
            self.results.update(top)

            self.queue.task_done()

            timestamp_end = time.time()
            log("Cost {:4f} secends to get the TOP{} cluster of {} is {}".format(timestamp_end-timestamp_start, self.n_top, test_id, top[test_id]), INFO)

class ProcessThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        threading.Thread.__init__(self, group=group, target=target, name=name, verbose=verbose)

        self.args = args

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.results = {}

    def get_nearest_centroid(self, filepath):
        df = pd.read_csv(filepath)

        for place_id in df["place_id"].unique():
            df_target = df[df["place_id"] == place_id]

            if self.is_exclude_outlier:
                df_target = df_target[(stats.zscore(df_target["x"]) < 3) & (stats.zscore(df_target["y"]) < 3) & (stats.zscore(df_target["accuracy"]) < 3)]

            x, y, accuracy = df_target["x"].mean(), df_target["y"].mean(), df_target["accuracy"].mean()

            yield place_id, (x, y, accuracy)

    def get_centroid_distance_metrics(self, filepath):
        filepath_pkl = "{}.centroid.metrics.isaccuracy={}_excludeoutiler={}.pkl".format(filepath, self.is_accuracy, self.is_exclude_outlier)

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

    def run(self):
        while True:
            timestamp_start = time.time()

            filepath_train = self.queue.get()
            filepath_test = filepath_train.replace("train", "test")
            filepath_pkl = filepath_train.replace("train", "prediction_tmp") + "isaccuracy={}_excludeoutlier={}.pkl".format(self.is_accuracy, self.is_exclude_outlier)
            filepath_submission = filepath_train.replace("train", "submission_tmp") + "isaccuracy={}_excludeoutlier={}.pkl".format(self.is_accuracy, self.is_exclude_outlier)
            for folder in [filepath_pkl, filepath_submission]:
                create_folder(folder)

            results = {}
            if not self.is_testing and os.path.exists(filepath_pkl):
                results = load_cache(filepath_pkl)
            else:
                centroid_metrics = self.get_centroid_distance_metrics(filepath_train)

                df = pd.read_csv(filepath_test)
                if self.is_testing:
                    df = df.head(100)
                log("Read testing data from {}".format(filepath_test), INFO)

                fields = ["x", "y"]
                if self.is_accuracy:
                    fields.append("accuracy")

                test_id = df["row_id"].values
                test_x = df[fields].values

                queue = Queue.Queue()
                for idx in range(0, test_id.shape[0]):
                    queue.put((test_id[idx], test_x[idx]))

                threads = []
                for idx in range(0, self.n_jobs/2):
                    thread = CalculateDistanceThread(kwargs={"queue": queue,
                                                             "centroid_metrics": centroid_metrics,
                                                             "distance": self.distance_function,
                                                             "n_top": self.n_top})
                    thread.setDaemon(True)
                    thread.start()

                    threads.append(thread)

                for thread in threads:
                    results.update(thread.results)

                queue.join()

            if not self.is_testing:
                save_submission(filepath_submission, results)
                save_cache(results, filepath_pkl)

            self.results.update(results)

            self.queue.task_done()

            timestamp_end = time.time()
            log("Cost {:8f} to finish the prediction of {}".format(timestamp_end-timestamp_start, filepath_train), INFO)

def process(workspace, is_accuracy, is_exclude_outlier, is_testing, n_top=3, n_jobs=8):
    results = {}

    queue = Queue.Queue()

    for filepath_train in glob.iglob(os.path.join(workspace, "*.csv")):
        queue.put(filepath_train)

    threads = []
    for idx in range(0, n_jobs):
        thread = ProcessThread(kwargs={"queue": queue,
                                       "distance_function": euclidean,
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
        results.update(thread.results)

    return results

def save_submission(filepath, results):
    pd.DataFrame(results.items(), columns=["row_id", "place_id"]).to_csv(filepath, index=False, compression="gzip")

if __name__ == "__main__":
    workspace = "/Users/RungChiChen/Documents/programs/kaggle/cases/Facebook V - Predicting Check Ins/input/1_way/train/pos/windown_size=2"

    is_accuracy, is_exclude_outlier, is_testing = False, False, True

    filepath_train = os.path.join(workspace, "train.csv")
    filepath_test = os.path.join(workspace, "test.csv")
    filepath_output = "{}.{}_isaccuracy={}_excludeoutlier={}.submission.csv".format(workspace, datetime.datetime.now().strftime("%Y%m%d%H"), is_accuracy, is_exclude_outlier)

    results = process(workspace, is_accuracy, is_exclude_outlier, is_testing)

    save_submission(filepath_output, results)

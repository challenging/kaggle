#!/usr/bin/env python

import os
import sys
import glob
import time
import datetime

import pandas as pd
import numpy as np

from joblib import Parallel, delayed

from scipy import stats
from scipy.spatial.distance import euclidean

from sklearn.neighbors import NearestCentroid, DistanceMetric

from utils import log, INFO
from load import save_cache, load_cache

def _get_nearest_centroid(df, place_id, is_exclude_outlier):
    df_target = df[df["place_id"] == place_id]

    if is_exclude_outlier:
        df_target = df_target[(stats.zscore(df_target["x"]) < 3) & (stats.zscore(df_target["y"]) < 3) & (stats.zscore(df_target["accuracy"]) < 3)]

    x, y, accuracy = df_target["x"].mean(), df_target["y"].mean(), df_target["accuracy"].mean()

    return place_id, (x, y, accuracy)

def get_nearest_centroid(filepath, is_exclude_outlier=False, n_jobs=8):
    df = pd.read_csv(filepath)

    place_ids = df["place_id"].unique()
    pool = Parallel(n_jobs=n_jobs)(delayed(_get_nearest_centroid)(df, place_id, is_exclude_outlier) for place_id in place_ids)

    for place_id, metrics in pool:
        yield place_id, metrics

def get_centroid_distance_metrics(filepath, is_accuracy=True, is_exclude_outlier=False):
    filepath_pkl = "{}.centroid.metrics.isaccuracy={}_excludeoutiler={}.pkl".format(filepath, is_accuracy, is_exclude_outlier)

    metrics = {}

    timestamp_start = time.time()
    if os.path.exists(filepath_pkl):
        metrics = load_cache(filepath_pkl)
    else:
        for place_id, m in get_nearest_centroid(filepath, is_exclude_outlier):
            metrics[m if is_accuracy else m[:2]] = place_id
    timestamp_end = time.time()

    log("Cost {:4} seconds to build up the centroid metrics".format(timestamp_end-timestamp_start), INFO)

    save_cache(metrics, filepath_pkl)

    return metrics

def calculate_distance(test_id, test_x, centroid_metrics, n_top=3, func=euclidean):
    timestamp_start = time.time()

    results = {}
    for metrics, place_id in centroid_metrics.items():
        results[place_id] = func(test_x, list(metrics))

    top = {test_id: []}
    for place_id, distance in sorted(results.items(), key=lambda (k, v): v)[:n_top]:
        top[test_id].append(str(place_id))
    top[test_id] = " ".join(top[test_id])

    timestamp_end = time.time()
    log("Cost {:4f} secends to get the TOP{} cluster of {} is {}".format(timestamp_end-timestamp_start, n_top, test_id, top[test_id]), INFO)

    return top

def prediction(filepath_train, filepath_test, is_accuracy, is_exclude_outliner, is_testing, n_top=3, n_jobs=8):
    centroid_metrics = get_centroid_distance_metrics(filepath_train, is_exclude_outliner)

    df = pd.read_csv(filepath_test)
    if is_testing:
        df = df.head(100)
    log("Read testing data from {}".format(filepath_test), INFO)

    fields = ["x", "y"]
    if is_accuracy:
        fields.append("accuracy")

    test_id = df["row_id"].values
    test_x = df[fields].values

    results = {}
    pool = Parallel(n_jobs=n_jobs)(delayed(calculate_distance)(test_id[idx], test_x[idx], centroid_metrics, n_top, euclidean) for idx in range(0, test_id.shape[0]))

    for sub_result in pool:
        results.update(sub_result)

    return results

def process(workspace, is_accuracy, is_exclude_outlier, is_testing):
    results = {}

    for filepath_train in glob.iglob(os.path.join(workspace, "*.csv")):
        filepath_test = filepath_train.replace("train", "test")
        log("Currently, working in {} and {}".format(filepath_train, filepath_test))

        results.update(prediction(filepath_train, filepath_test, is_accuracy, is_exclude_outlier, is_testing, n_top=3, n_jobs=16))

    return results

def save_submission(filepath, results):
    pd.DataFrame(results.items(), columns=["row_id", "place_id"]).to_csv(filepath, index=False, compression="gzip")

if __name__ == "__main__":
    workspace = "/Users/RungChiChen/Documents/programs/kaggle/cases/Facebook V - Predicting Check Ins/input/1_way/train/pos/windown_size=1"

    filepath_train = os.path.join(workspace, "train.csv")
    filepath_test = os.path.join(workspace, "test.csv")
    filepath_output = "{}.{}.submission.csv".format(workspace, datetime.datetime.now().strftime("%Y%m%d%H"))

    is_accuracy, is_exclude_outlier, is_testing = False, False, False
    results = process(workspace, is_accuracy, is_exclude_outlier, is_testing)

    save_submission(filepath_output, results)

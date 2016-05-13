#!/usr/bin/env python

import os
import sys
import datetime

import pandas as pd
import numpy as np

from joblib import Parallel, delayed

from scipy import stats
from scipy.spatial.distance import euclidean

from sklearn.neighbors import NearestCentroid, DistanceMetric

from utils import log, INFO

def _get_nearest_centroid(df, place_id, is_exclude_outlier):
    df_target = df[df["place_id"] == place_id]

    if is_exclude_outlier:
        df_target = df_target[(stats.zscore(df_target["x"]) < 3) & (stats.zscore(df_target["y"]) < 3) & (stats.zscore(df_target["accuracy"]) < 3)]

    x, y, accuracy = df_target["x"].mean(), df_target["y"].mean(), df_target["accuracy"].mean()

    log("Get the centroid of {} is ({},{},{})".format(place_id, x, y, accuracy), INFO)
    return place_id, (x, y, accuracy)

def get_nearest_centroid(filepath, is_exclude_outlier=False, n_jobs=8):
    df = pd.read_csv(filepath)

    place_ids = df["place_id"].unique()
    pool = Parallel(n_jobs=n_jobs)(delayed(_get_nearest_centroid)(df, place_id, is_exclude_outlier) for place_id in place_ids)

    for place_id, metrics in pool:
        yield place_id, metrics

def get_centroid_distance_metrics(filepath, is_accuracy=True, is_exclude_outliner=False):
    metrics = {}

    for place_id, m in get_nearest_centroid(filepath, is_exclude_outliner):
        metrics[m if is_accuracy else m[:2]] = place_id

    return metrics

def calculate_distance(test_id, test_x, centroid_metrics, n_top=3, func=euclidean):
    results = {}

    for metrics, place_id in centroid_metrics.items():
        results[place_id] = func(test_x, list(metrics))

    top = {test_id: []}
    for place_id, distance in sorted(results.items(), key=lambda (k, v): v)[:n_top]:
        top[test_id].append(str(place_id))
    top[test_id] = " ".join(top[test_id])

    log("Get the TOP{} cluster of {} is {}".format(n_top, test_id, top[test_id]), INFO)

    return top

def prediction(filepath_train, filepath_test, is_accuracy, is_exclude_outliner, is_testing, n_top=3, n_jobs=4):
    centroid_metrics = get_centroid_distance_metrics(filepath_train, is_exclude_outliner)

    df = pd.read_csv(filepath_test)
    if is_testing:
        df = df.head(100)

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

def save_submission(filepath, results):
    pd.DataFrame(results.items(), columns=["row_id", "place_id"]).to_csv(filepath, index=False)

if __name__ == "__main__":
    workspace = "/Users/RungChiChen/Documents/programs/kaggle/cases/Facebook V - Predicting Check Ins/input"
    filepath_train = os.path.join(workspace, "train.csv")
    filepath_test = os.path.join(workspace, "test.csv")
    filepath_output = os.path.join(workspace, datetime.datetime.now().strftime("%Y%m%d%H"), "submission.csv")

    is_accuracy, is_exclude_outlier, is_testing = False, False, False
    results = prediction(filepath_train, filepath_test, is_accuracy, is_exclude_outlier, is_testing, n_top=3, n_jobs=16)

    save_submission(filepath_output, results)

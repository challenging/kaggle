#!/usr/bin/env python

import os
import sys
import pprint

import click
import numpy as np
import pandas as pd

import feature_engineering

from utils import create_folder, log, INFO, WARN, ERROR
from load import load_data, data_transform_2, load_interaction_information, save_cache, load_cache
from parameter_tuning import XGBoostingTuning, RandomForestTuning, ExtraTreeTuning, tuning
from feature_engineering import pca
from configuration import ModelConfParser

@click.command()
@click.option("--conf", required=True, help="Filepath of Configuration")
@click.option("--n-jobs", default=1, help="Number of thread")
@click.option("--is-pca", is_flag=True, help="Turn on the PCA mode")
@click.option("--is-testing", is_flag=True, help="Testing mode")
@click.option("--methodology", required=True, help="Tune parameters of which methodology")
@click.option("--nfold", default=3, help="the number of nfold")
@click.option("--n-estimator", default=200, help="the number of estimator")
@click.option("--dropout", default=0, help="dropout rate")
def parameter_tuning(methodology, nfold, is_pca, is_testing, n_jobs, conf, n_estimator, dropout):
    drop_fields = []

    parser = ModelConfParser(conf)

    objective = parser.get_objective()
    cost = parser.get_cost()

    filepath_training, filepath_testing, filepath_submission, filepath_tuning = parser.get_filepaths(methodology)
    filepath_feature_importance, top_feature = None, None

    for filepath in [filepath_tuning, filepath_submission]:
        create_folder(filepath)

    df_training = pd.read_csv(filepath_training)
    df_testing = pd.read_csv(filepath_testing)

    eps = 0.00001

    original_size = df_training.shape[0]
    df_training = df_training.groupby("place_id").filter(lambda x: len(x) >= dropout)
    print ("Before: %d rows || After: %d rows" % (original_size, df_training.shape[0]))

    initial_date = np.datetime64('2014-01-01T01:01', dtype='datetime64[m]')
    d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn), 'm') for mn in df_training["time"].values)

    train_x = df_training[["x", "y", "accuracy", "time"]]
    train_x["hourofday"] = d_times.hour
    train_x["dayofmonth"] = d_times.day
    train_x["weekday"] = d_times.weekday
    train_x["monthofyear"] = d_times.month
    train_x["year"] = d_times.year

    train_x = train_x.drop(["time"], axis=1)
    train_y = df_training["place_id"].astype(str)

    values, counts = np.unique(train_y, return_counts=True)

    d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn), 'm') for mn in df_testing["time"].values)

    test_x = df_testing[["x", "y", "accuracy", "time"]]
    test_x["hourofday"] = d_times.hour
    test_x["dayofmonth"] = d_times.day
    test_x["weekday"] = d_times.weekday
    test_x["monthofyear"] = d_times.month
    test_x["year"] = d_times.year
    test_x = test_x.drop(["time"], axis=1)

    pool = []
    for value, count in zip(values, counts):
        if count > 2:
            pool.append(value)

    idxs = train_y.isin(pool)

    train_x = train_x[idxs].values
    train_y = train_y[idxs].astype(str).values

    test_x = test_x.values
    test_id = df_testing["row_id"].values

    if is_testing:
        train_x = train_x.head(1000)
        train_y = train_y.head(1000)

    params = tuning(train_x, train_y, test_id, test_x, cost, objective,
                    filepath_feature_importance, filepath_tuning, filepath_submission, methodology, nfold, top_feature,
                    n_estimator=n_estimator, thread=n_jobs, is_saving=False)

    log("The final parameters are {}".format(params))

if __name__ == "__main__":
    parameter_tuning()

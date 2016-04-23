#!/usr/bin/env python

import os
import sys
import pprint

import click
import numpy as np
import pandas as pd

BASEPATH = os.path.dirname(os.path.abspath(__file__))

sys.path.append("{}/../lib".format(BASEPATH))
import feature_engineering

from utils import log, INFO, WARN, ERROR
from load import load_data, data_transform_2, load_interaction_information, save_cache, load_cache
from parameter_tuning import XGBoostingTuning, RandomForestTuning, ExtraTreeTuning
from configuration import ModelConfParser

@click.command()
@click.option("--conf", required=True, help="Filepath of Configuration")
@click.option("--thread", default=1, help="Number of thread")
@click.option("--is-testing", is_flag=True, help="Testing mode")
@click.option("--is-feature-importance", is_flag=True, help="Turn on the feature importance")
@click.option("--methodology", required=True, help="Tune parameters of which methodology")
@click.option("--nfold", default=5, help="the number of nfold")
def tuning(methodology, nfold, is_testing, is_feature_importance, thread, conf):
    drop_fields = []
    N = 650 - len(drop_fields)

    parser = ModelConfParser(conf)
    BASEPATH = parser.get_workspace()
    n_jobs = parser.get_n_jobs()
    cost = parser.get_cost()
    binsize, top = parser.get_interaction_information()
    top_feature = parser.get_top_feature()

    filepath_training = "{}/input/train.csv".format(BASEPATH)
    filepath_testing = "{}/input/test.csv".format(BASEPATH)
    filepath_cache_1 = "{}/input/{}_training_dataset.cache".format(BASEPATH, N)
    folder_ii = "{}/input/interaction_information/transform2=True_testing=-1_binsize={}".format(BASEPATH, binsize)
    filepath_tuning = "{}/etc/parameter_tuning/{}_testing={}_nfold={}_top={}_binsize={}.pkl".format(BASEPATH, methodology, is_testing, nfold, top, binsize)
    filepath_feature_importance = "{}/etc/feature_profile/transform2=True_binsize={}_top={}.pkl".format(BASEPATH, binsize, top)

    train_x = None
    train_x, test_x, train_y, test_id, train_id = load_data(filepath_cache_1, filepath_training, filepath_testing, drop_fields)

    for layers, value in load_interaction_information(folder_ii, top):
        for df in [train_x, test_x]:
            t = value
            breaking_layer = None
            for layer in layers:
                if layer in train_x.columns:
                    t *= df[layer]
                else:
                    breaking_layer = layer
                    break

            if breaking_layer == None:
                df[";".join(layers)] = t
            else:
                log("Skip {}".format(layers), WARN)
                break

    train_x["Target"] = train_y.values

    if is_testing:
        train_x = train_x.head(1000)

    log("{} data records with {} features".format(len(train_x), len(train_x.columns)))

    algorithm, is_classifier = None, False
    if methodology.find("xg") > -1:
        if methodology[-1] == "c":
            algorithm = XGBoostingTuning("Target", "ID", "classifier", cost=cost, n_jobs=thread, cv=nfold)

            is_classifier = True
        elif methodology[-1] == "r":
            algorithm = XGBoostingTuning("Target", "ID", "regressor", cost=cost, n_jobs=thread, cv=nfold)
    elif methodology.find("rf") > -1:
        if methodology[-1] == "c":
            algorithm = RandomForestTuning("Target", "ID", "classifier", cost=cost,n_jobs=thread, cv=nfold)

            is_classifier = True
        elif methodology[-1] == "r":
            algorithm = RandomForestTuning("Target", "ID", "regressor", cost=cost,n_jobs=thread, cv=nfold)
    elif methodology.find("et") > -1:
        if methodology[-1] == "c":
            algorithm = ExtraTreeTuning("Target", "ID", "classifier", cost=cost,n_jobs=thread, cv=nfold)

            is_classifier = True
        elif methodology[-1] == "r":
            algorithm = ExtraTreeTuning("Target", "ID", "regressor", cost=cost,n_jobs=thread, cv=nfold)

    if algorithm == None:
        log("Not support this algorithm - {}".format(methodology), ERROR)
        sys.exit(1)

    algorithm.set_train(train_x)
    if is_classifier:
        algorithm.enable_feature_importance(filepath_feature_importance, top_feature)

    algorithm.set_filepath(filepath_tuning)

    if os.path.exists(filepath_tuning):
        algorithm.load()

    if is_feature_importance:
        algorithm.enable_feature_importance()

    algorithm.process()

if __name__ == "__main__":
    tuning()

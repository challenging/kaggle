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

from utils import log, INFO
from load import load_data, data_transform_2, load_interaction_information, save_cache, load_cache
from parameter_tuning import XGBoostingTuning, RandomForestTuning, ExtraTreeTuning
from configuration import ModelConfParser

@click.command()
@click.option("--conf", required=True, help="Filepath of Configuration")
@click.option("--thread", default=1, help="Number of thread")
@click.option("--is-testing", is_flag=True, help="Testing mode")
@click.option("--methodology", required=True, help="Tune parameters of which methodology")
@click.option("--binsize", default=16, help="bin/bucket size setting")
@click.option("--combinations-size", default=2, help="size of combinations")
@click.option("--top", default=300, help="Extract how many interaction information we extract")
def tuning(methodology, binsize, combinations_size, top, is_testing, thread, conf):
    drop_fields = []
    N = 650 - len(drop_fields)

    parser = ModelConfParser(conf)
    BASEPATH = parser.get_workspace()
    n_jobs = parser.get_n_jobs()

    filepath_training = "{}/input/train.csv".format(BASEPATH)
    filepath_testing = "{}/input/test.csv".format(BASEPATH)
    filepath_cache_1 = "{}/input/{}_training_dataset.cache".format(BASEPATH, N)
    filepath_ii = "{}/input/transform2=True_testing=-1_type=2_binsize={}_combination={}.pkl".format(BASEPATH, binsize, combinations_size)
    filepath_tuning = "{}/etc/parameter_tuning/{}_testing={}_binsize={}_combination={}.pkl".format(BASEPATH, methodology, is_testing, binsize, combinations_size)

    train_x = None
    train_x, test_x, train_y, test_id, train_id = load_data(filepath_cache_1, filepath_training, filepath_testing, drop_fields)

    for (layer1, layer2), value in load_interaction_information(filepath_ii, top):
        train_x["{}-{}".format(layer1, layer2)] = train_x[layer1].values * train_x[layer2].values * value
        test_x["{}-{}".format(layer1, layer2)] = test_x[layer1].values * test_x[layer2].values * value

    train_x["Target"] = train_y.values
    train_x = train_x.head(int(len(train_x)*0.9))

    if is_testing:
        train_x = train_x.head(1000)

    log("{} data records with {} features".format(len(train_x), len(train_x.columns)))

    algorithm = None
    if methodology.find("xg") > -1:
        if methodology[-1] == "c":
            algorithm = XGBoostingTuning("Target", "ID", "classifier", n_jobs=thread)
        elif methodology[-1] == "r":
            algorithm = XGBoostingTuning("Target", "ID", "regressor", n_jobs=thread)
    elif methodology.find("rf") > -1:
        if methodology[-1] == "c":
            algorithm = RandomForestTuning("Target", "ID", "classifier", n_jobs=thread)
        elif methodology[-1] == "r":
            algorithm = RandomForestTuning("Target", "ID", "regressor", n_jobs=thread)
    elif methodology.find("et") > -1:
        if methodology[-1] == "c":
            algorithm = ExtraTreeTuning("Target", "ID", "classifier", n_jobs=thread)
        elif methodology[-1] == "r":
            algorithm = ExtraTreeTuning("Target", "ID", "regressor", n_jobs=thread)

    algorithm.set_train(train_x)
    algorithm.set_filepath(filepath_tuning)

    if os.path.exists(filepath_tuning):
        algorithm.load()

    algorithm.process()

if __name__ == "__main__":
    tuning()

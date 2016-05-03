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
from parameter_tuning import XGBoostingTuning, RandomForestTuning, ExtraTreeTuning, tuning
from configuration import ModelConfParser

@click.command()
@click.option("--conf", required=True, help="Filepath of Configuration")
@click.option("--thread", default=1, help="Number of thread")
@click.option("--is-testing", is_flag=True, help="Testing mode")
@click.option("--is-feature-importance", is_flag=True, help="Turn on the feature importance")
@click.option("--methodology", required=True, help="Tune parameters of which methodology")
@click.option("--nfold", default=5, help="the number of nfold")
def parameter_tuning(methodology, nfold, is_testing, is_feature_importance, thread, conf):
    drop_fields = []

    parser = ModelConfParser(conf)
    BASEPATH = parser.get_workspace()
    n_jobs = parser.get_n_jobs()
    cost = parser.get_cost()
    binsize, top = parser.get_interaction_information()
    top_feature = parser.get_top_feature()

    filepath_training = "{}/input/train.csv".format(BASEPATH)
    filepath_testing = "{}/input/test.csv".format(BASEPATH)
    filepath_cache_1 = "{}/input/train.pkl".format(BASEPATH)
    folder_ii = "{}/input/interaction_information/transform2=True_testing=-1_binsize={}".format(BASEPATH, binsize)
    filepath_feature_importance = "{}/etc/feature_profile/transform2=True_binsize={}_top={}.pkl".format(BASEPATH, binsize, top)
    filepath_submission = "{}/etc/parameter_tuning/{}_transform2=True_binsize={}_top={}_topfeature={}.submission.csv".format(BASEPATH, methodology, binsize, top, top_feature)

    parent_folder = os.path.dirname(filepath_submission)
    if not os.path.isdir(parent_folder):
        os.makedirs(parent_folder)

    train_x = None
    train_x, test_x, train_y, test_id, train_id = load_data(filepath_cache_1, filepath_training, filepath_testing, drop_fields)

    for layers, value in load_interaction_information(folder_ii, str(top_feature)):
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

    if is_testing:
        train_x = train_x.head(1000)
        train_y = train_y.head(1000)

    filepath_tuning = "{}/etc/parameter_tuning/{}_testing={}_nfold={}_top={}_binsize={}_feature={}.pkl".format(BASEPATH, methodology, is_testing, nfold, top_feature, binsize, len(train_x.columns))
    log("{} data records with {} features, and filepath is {}".format(len(train_x), len(train_x.columns), filepath_tuning), INFO)

    params = tuning(train_x, train_y, test_id, test_x, cost,
                    filepath_feature_importance if is_feature_importance else None, filepath_tuning, filepath_submission, methodology, nfold, top_feature, binsize,
                    thread=thread)

    log("The final parameters are {}".format(params))

if __name__ == "__main__":
    parameter_tuning()

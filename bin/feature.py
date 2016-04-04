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
from load import data_load, data_transform_2, save_cache, load_cache

@click.command()
@click.option("--thread", default=1, help="Number of thread")
@click.option("--transform2", is_flag=True, help="Transform source data by style-2")
@click.option("--feature-importance", is_flag=True, help="Calculate the feature importance")
@click.option("--interaction-information", is_flag=True, help="Calculate the interaction information")
@click.option("--binsize", default=16, help="bin/bucket size setting")
@click.option("--testing", default=-1, help="cut off the input file to be the testing dataset")
def feature_engineer(thread, transform2, feature_importance, interaction_information, binsize, testing):
    drop_fields = []

    if feature_importance:
        log("Try to calculate the feature ranking/score/importance", INFO)
        train_x, train_y, test_x, test_id = data_load(drop_fields=drop_fields)
        if transform2:
            train_x, test_x = data_transform_2(train_x, test_x)

        names = train_x.columns
        print "Data Distribution is ({}, {}), and then the number of feature is {}".format(np.sum(train_y==0), np.sum(train_y==1), len(names))

        # output folder
        folder_feature = "{}/../feature_profiling".format(BASEPATH)
        if not os.path.isdir(folder_feature):
            os.makedirs(folder_feature)

        names = list(train_x.columns.values)
        filepath_feature = "{}/BNP.csv".format(folder_feature)

        fp = feature_engineering.FeatureProfile()
        ranks = fp.profile(train_x.values, train_y, names, filepath_feature, int(len(train_x.columns)*0.5))

    if interaction_information:
        log("Try to calculate the interaction information", INFO)

        filepath_training = "{}/../input/train.csv".format(BASEPATH)
        filepath_testing = "{}/../input/test.csv".format(BASEPATH)
        filepath_cache = "{}/../input/transform2={}_cache.pkl".format(BASEPATH, transform2)

        train_x, test_x, train_y, id_train, id_test = None, None, None, None, None
        if transform2:
            train_x, test_x, train_y, id_train, id_test = data_transform_2(filepath_training, filepath_testing, keep_nan=True)
        else:
            train_x, train_y, test_x, test_id = data_load(drop_fields=drop_fields)

        filepath_couple = "{}/../input/transform2={}_testing={}_type=2_binsize={}.pkl".format(BASEPATH, transform2, testing, binsize)
        filepath_single = "{}/../input/transform2={}_testing={}_type=1_binsize={}.pkl".format(BASEPATH, transform2, testing, binsize)

        results_single, results_couple = feature_engineering.calculate_interaction_information(filepath_cache,\
            train_x, train_y, filepath_couple, filepath_single,\
            binsize=binsize, nthread=thread, threshold=0.01, is_testing=int(testing) if testing != -1 else None)

if __name__ == "__main__":
    feature_engineer()

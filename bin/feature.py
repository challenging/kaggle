#!/usr/bin/env python

import os
import sys

import click
import numpy as np
import pandas as pd

BASEPATH = os.path.dirname(os.path.abspath(__file__))

sys.path.append("{}/../lib".format(BASEPATH))
import feature_engineering

from utils import log, INFO
from load import data_load, data_transform_2

@click.command()
@click.option("--feature-importance", is_flag=True, help="Calculate the feature importance")
@click.option("--interaction-information", is_flag=True, help="Calculate the interaction information")
@click.option("--binsize", default=16, help="bin/bucket size setting")
def feature_engineer(feature_importance, interaction_information, binsize):
    if feature_importance:
        log("Try to calculate the feature ranking/score/importance", INFO)
        drop_fields = []
        train_x, train_y, test_x, test_id = data_load(drop_fields=drop_fields)
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

        train_x, test_x, train_y, id_train, id_test = data_transform_2(filepath_training, filepath_testing, keep_nan=True)

        results_single, results_couple = feature_engineering.interaction_information(train_x, train_y, binsize=binsize, threshold=0.01)
        results = results_single
        results.update(results_couple)

        filepath_results = "{}/../input/transform2_binsize={}.csv".format(BASEPATH, binsize)
        pd.DataFrame(results).to_csv(filepath_results)

if __name__ == "__main__":
    feature_engineer()

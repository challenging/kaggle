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

from interaction_information import FeatureProfile
from utils import log, INFO
from load import data_load, load_data, data_transform_2, load_interaction_information, save_cache, load_cache
from configuration import ModelConfParser

@click.command()
@click.option("--conf", required=True, help="Filepath of Configuration")
@click.option("--thread", default=1, help="Number of thread")
@click.option("--feature-importance", is_flag=True, help="Calculate the feature importance")
@click.option("--interaction-information", is_flag=True, help="Calculate the interaction information")
@click.option("--merge-ii", is_flag=True, help="merge the pickle files of interaction information")
@click.option("--split-idx", default=0, help="the index of split number")
@click.option("--split-num", default=1, help="the split number of combinations-size")
@click.option("--testing", default=-1, help="cut off the input file to be the testing dataset")
@click.option("--combinations-size", default=3, help="size of combinations")
def feature_engineer(conf, thread, feature_importance, interaction_information, merge_ii, split_idx, split_num, testing, combinations_size):
    drop_fields = []

    N, transform2 = 650, True

    parser = ModelConfParser(conf)
    BASEPATH = parser.get_workspace()
    binsize, top = parser.get_interaction_information()

    if feature_importance:
        filepath_training = "{}/input/train.csv".format(BASEPATH)
        filepath_testing = "{}/input/test.csv".format(BASEPATH)
        filepath_cache_1 = "{}/input/{}_training_dataset.cache".format(BASEPATH, N)
        folder_ii = "{}/input/interaction_information/transform2=True_testing=-1_binsize={}".format(BASEPATH, binsize)

        train_x, test_x, train_y, test_id, train_id = load_data(filepath_cache_1, filepath_training, filepath_testing, drop_fields)

        columns = train_x.columns
        for layers, value in load_interaction_information(folder_ii, threshold=top):
            for df in [train_x, test_x]:
                t = value
                breaking_layer = None
                for layer in layers:
                    if layer in columns:
                        t *= df[layer].values
                    else:
                        breaking_layer = layer
                        break
                if breaking_layer == None:
                    df[";".join(layers)] = t
                else:
                    log("Skip {} due to {} not in columns".format(layers, breaking_layer), WARN)
                    break

        names = train_x.columns
        print "Data Distribution is ({}, {}), and then the number of feature is {}".format(np.sum(train_y==0), np.sum(train_y==1), len(names))

        # output folder
        folder_feature = "{}/etc/feature_profile".format(BASEPATH)
        if not os.path.isdir(folder_feature):
            os.makedirs(folder_feature)

        names = list(train_x.columns.values)
        folder_feature = "{}/transform2=True_binsize={}_top={}".format(folder_feature, binsize, top)

        fp = FeatureProfile()
        fp.profile(train_x.values, train_y, names, folder_feature, int(len(train_x.columns)*0.5))

    if interaction_information:
        log("Try to calculate the interaction information", INFO)

        filepath_training = "{}/input/train.csv".format(BASEPATH)
        filepath_testing = "{}/input/test.csv".format(BASEPATH)

        train_x, test_x, train_y, id_train, id_test = None, None, None, None, None
        if transform2:
            train_x, test_x, train_y, id_train, id_test = data_transform_2(filepath_training, filepath_testing, keep_nan=True)
        else:
            train_x, train_y, test_x, test_id = data_load(drop_fields=drop_fields)

        filepath_cache = "{}/input/transform2={}_binsize={}_cache.pkl".format(BASEPATH, transform2, binsize)
        folder_couple = "{}/input/interaction_information/transform2={}_testing={}_binsize={}".format(BASEPATH, transform2, testing, binsize)

        results_couple = feature_engineering.calculate_interaction_information(filepath_cache, train_x, train_y, folder_couple, \
            binsize=binsize, nthread=thread, combinations_size=combinations_size, n_split_idx=split_idx, n_split_num=split_num,
            is_testing=int(testing) if testing > 0 else None)

    if merge_ii:
        folder_couple = "{}/input/interaction_information/transform2={}_testing={}_binsize={}".format(BASEPATH, transform2, testing, binsize)

        count_filepath, count_couple, final_count_filepath, final_count_couple = feature_engineering.merge_interaction_information(folder_couple)
        log("Originally. we have {} records in {} files. After merging, we have {} records in {} files".format(count_couple, count_filepath, final_count_couple, final_count_filepath), INFO)

if __name__ == "__main__":
    feature_engineer()

#!/usr/bin/env python

import os
import sys
import click

import numpy as np
import pandas as pd

sys.path.append("{}/../lib".format(os.path.dirname(os.path.abspath(__file__))))
from load import data_load, data_transform_1, data_transform_2, pca
from feature import FeatureProfile

BASEPATH = os.path.dirname(os.path.abspath(__file__))

@click.command()
def feature():
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

    fp = FeatureProfile()
    ranks = fp.profile(train_x.values, train_y, names, filepath_feature, int(len(train_x.columns)*0.5))

if __name__ == '__main__':
    feature()

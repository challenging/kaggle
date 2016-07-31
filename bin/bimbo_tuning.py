#!/usr/bin/env python

import os
import click

import numpy as np
import pandas as pd

from utils import create_folder, log, INFO
from parameter_tuning import tuning
from configuration import ModelConfParser

from bimbo.constants import SPLIT_PATH
from bimbo.constants import COLUMN_WEEK, COLUMN_ROW, COLUMNS, MONGODB_COLUMNS

@click.command()
@click.option("--n-jobs", default=1, help="Number of thread")
@click.option("--methodology", required=True, help="Tune parameters of which methodology")
@click.option("--option", required=True, nargs=2, type=click.Tuple([unicode, unicode]), default=(None, None), help="ex. route_id 1118")
@click.option("--nfold", default=3, help="the number of nfold")
@click.option("--n-estimator", default=50, help="the number of estimator")
def parameter_tuning(methodology, nfold, n_jobs, n_estimator, option):
    column, fileid = option

    is_saving = False
    filepath_feature_importance, top_feature = None, None
    filepath_submission, filepath_tuning = os.path.join("tmp", "submission", COLUMNS[column], fileid), os.path.join("tmp", "tuning", COLUMNS[column], fileid)
    cost, objective = "rmsle", "reg:linear"

    filepath = os.path.join(SPLIT_PATH + ".numeric", COLUMNS[column], "train", "{}.csv".format(fileid))

    df_training = pd.read_csv(filepath)

    df_testing = df_training[df_training[COLUMN_WEEK] == 9].copy()

    df_training.drop([COLUMN_WEEK], axis=1, inplace=True)
    df_testing.drop([COLUMN_WEEK], axis=1, inplace=True)

    y_column = ["Demanda_uni_equil"]

    x_columns = df_training.columns.values
    x_columns = np.delete(x_columns, np.where(x_columns == y_column[0]))

    train_x = df_training[x_columns]
    train_y = df_training[y_column]
    log("The shape of x,y are {},{}".format(train_x.shape, train_y.shape), INFO)

    test_id = np.array([[i] for i in range(0, df_testing.shape[0])])
    test_x = df_testing[x_columns]
    log("The shape of test_x is {}/{}".format(test_id.shape, test_x.shape), INFO)

    params = tuning(train_x, train_y, test_id, test_x, cost, objective,\
                    filepath_feature_importance, filepath_tuning, filepath_submission, methodology, nfold, top_feature,\
                    n_estimator=n_estimator, thread=n_jobs, is_saving=is_saving)

    log("The final parameters are {}".format(params), INFO)

if __name__ == "__main__":
    parameter_tuning()

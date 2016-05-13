#!/usr/bin/env python

import os
import sys
import datetime

import click

BASEPATH = os.path.dirname(os.path.abspath(__file__))

sys.path.append("{}/../lib".format(BASEPATH))

from utils import log, INFO
from facebook_learning import process, save_submission
from configuration import FacebookConfiguration

@click.command()
@click.option("--conf", required=True, help="Filepath of Configuration")
@click.option("--n-jobs", default=4, help="Number of thread")
@click.option("--is-testing", is_flag=True, help="Testing Mode")
@click.option("--n-top", default=10, help="Top X cluster")
def facebook(conf, n_jobs, is_testing, n_top):
    configuration = FacebookConfiguration(conf)

    workspace = configuration.get_workspace()
    is_accuracy, is_exclude_outlier = configuration.is_accuracy(), configuration.is_exclude_outlier()

    filepath_train = os.path.join(workspace, "train.csv")
    filepath_test = os.path.join(workspace, "test.csv")
    filepath_output = "{}/{}_isaccuracy={}_excludeoutlier={}.submission.csv.gz".format(workspace, datetime.datetime.now().strftime("%Y%m%d%H"), is_accuracy, is_exclude_outlier)

    results = process(workspace, is_accuracy, is_exclude_outlier, is_testing, n_top=n_top, n_jobs=n_jobs)

    save_submission(filepath_output, results)

if __name__ == "__main__":
    facebook()

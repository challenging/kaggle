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
def facebook(conf, n_jobs, is_testing):
    configuration = FacebookConfiguration(conf)

    workspace, cache_workspace, output_workspace = configuration.get_workspace()
    is_accuracy, is_exclude_outlier = configuration.is_accuracy(), configuration.is_exclude_outlier()
    window_size, batch_size, n_top = configuration.get_size()
    method, criteria = configuration.get_method()
    stamp = configuration.get_stamp()

    log("The method is {}, window_size is {}, batch_size is {}. n_top is {}".format(method, window_size, batch_size, n_top))

    filepath_train = os.path.join(workspace, "train.csv")
    filepath_test = os.path.join(workspace, "test.csv")
    cache_workspace = "{}/method={},{}_windowsize={}_batchsize={}_isaccuracy={}_excludeoutlier={}_istesting={}/{}.{}".format(\
                        cache_workspace, method, "x".join(criteria), window_size, batch_size, is_accuracy, is_exclude_outlier, is_testing, stamp, n_top)
    submission_workspace = "{}/method={},{}_windowsize={}_batchsize={}_isaccuracy={}_excludeoutlier={}_istesting={}/{}.{}".format(\
                        output_workspace, method, "x".join(criteria), window_size, batch_size, is_accuracy, is_exclude_outlier, is_testing, stamp, n_top)

    log("The workspace is {}".format(workspace))
    log("The cache workspace is {}".format(cache_workspace), INFO)
    log("The submission workspace is {}".format(submission_workspace), INFO)

    results = process(method, (workspace, cache_workspace, submission_workspace), batch_size, criteria, is_accuracy, is_exclude_outlier, is_testing, n_top=n_top, n_jobs=n_jobs)

    filepath_output = submission_workspace + ".csv.gz"

    for size in [n_top, 3]:
        filepath_output = submission_workspace + ".{}.csv.gz".format(size)
        save_submission(filepath_output, results, size)

if __name__ == "__main__":
    facebook()

#!/usr/bin/env python

import os
import sys
import datetime

import pickle
import click

from load import load_cache
from utils import log, INFO
from facebook.facebook_learning import process, save_submission
from configuration import FacebookConfiguration

@click.command()
@click.option("--conf", required=True, help="Filepath of Configuration")
@click.option("--is-testing", is_flag=True, help="Turn on the testing mode")
def facebook_weight(conf, is_testing):
    configuration = FacebookConfiguration(conf)

    results = {}
    final_submission_filename = []
    for m in configuration.get_methods():
        workspace, cache_workspace, output_workspace = configuration.get_workspace(m)
        method, criteria, strategy, stamp, (window_size, batch_size, n_top), is_accuracy, is_exclude_outlier = configuration.get_method_detail(m)

        weight = configuration.get_weight(m)

        if method == "native":
            cache_workspace = "{}/criteria={}_windowsize={}_batchsize={}_isaccuracy={}_excludeoutlier={}_istesting={}/method={}.{}.{}".format(\
                                cache_workspace, criteria if isinstance(criteria, str) else "x".join(criteria), window_size, batch_size, is_accuracy, is_exclude_outlier, is_testing, method, stamp, n_top)
            submission_workspace = "{}/criteria={}_windowsize={}_batchsize={}_isaccuracy={}_excludeoutlier={}_istesting={}/method={}.{}.{}".format(\
                                output_workspace, criteria if isinstance(criteria, str) else "x".join(criteria), window_size, batch_size, is_accuracy, is_exclude_outlier, is_testing, method, stamp, n_top)
        else:
            cache_workspace = "{}/criteria={}_windowsize={}_batchsize={}_isaccuracy={}_excludeoutlier={}_istesting={}/method={}_strategy={}.{}.{}".format(\
                                cache_workspace, criteria if isinstance(criteria, str) else "x".join(criteria), window_size, batch_size, is_accuracy, is_exclude_outlier, is_testing, method, strategy, stamp, n_top)
            submission_workspace = "{}/criteria={}_windowsize={}_batchsize={}_isaccuracy={}_excludeoutlier={}_istesting={}/method={}_strategy={}.{}.{}".format(\
                                output_workspace, criteria if isinstance(criteria, str) else "x".join(criteria), window_size, batch_size, is_accuracy, is_exclude_outlier, is_testing, method, strategy, stamp, n_top)

        filepath_pkl = os.path.join(cache_workspace, "final_results.pkl")
        log("The filepath_pkl is {}".format(filepath_pkl), INFO)

        load_cache(filepath_pkl, is_hdb=True, others=(results, weight))

        final_submission_filename.append("-".join([stamp, str(weight)]))

    filepath_output = "{}.{}.csv".format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"), "_".join(final_submission_filename))
    save_submission(filepath_output, results, 3)

if __name__ == "__main__":
    facebook_weight()

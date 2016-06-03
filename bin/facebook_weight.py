#!/usr/bin/env python

import os
import sys
import datetime

import click

from load import load_cache
from utils import log, INFO
from utils import make_a_stamp
from facebook.facebook_learning import transform_to_submission_format, save_submission
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
        method, criteria, strategy, stamp, (window_size, batch_size, n_top), is_accuracy, is_exclude_outlier, is_normalization = configuration.get_method_detail(m)

        weight = configuration.get_weight(m)
        is_full = configuration.is_full()

        setting = configuration.get_setting("{}-SETTING".format(m))
        setting_stamp = make_a_stamp(setting)

        normalization = "normalization_" if is_normalization else ""
        grid_size = criteria if isinstance(criteria, str) else "x".join(criteria)

        if method == "native":
            cache_workspace = "{}/{}criteria={}_windowsize={}_batchsize={}_isaccuracy={}_excludeoutlier={}_istesting={}/method={}.{}.{}/{}".format(\
                cache_workspace, normalization, grid_size, window_size, batch_size, is_accuracy, is_exclude_outlier, is_testing, method, stamp, n_top, setting_stamp)
            submission_workspace = "{}/{}criteria={}_windowsize={}_batchsize={}_isaccuracy={}_excludeoutlier={}_istesting={}/{}/method={}.{}.{}".format(\
                output_workspace, normalization, grid_size, window_size, batch_size, is_accuracy, is_exclude_outlier, is_testing, setting_stamp, method, stamp, n_top)
        else:
            cache_workspace = "{}/{}criteria={}_windowsize={}_batchsize={}_isaccuracy={}_excludeoutlier={}_istesting={}/method={}_strategy={}.{}.{}/{}".format(\
                cache_workspace, normalization, grid_size, window_size, batch_size, is_accuracy, is_exclude_outlier, is_testing, method, strategy, stamp, n_top, setting_stamp)
            submission_workspace = "{}/{}criteria={}_windowsize={}_batchsize={}_isaccuracy={}_excludeoutlier={}_istesting={}/{}/method={}_strategy={}.{}.{}".format(\
                output_workspace, normalization, grid_size, window_size, batch_size, is_accuracy, is_exclude_outlier, is_testing, setting_stamp, method, strategy, stamp, n_top)

        log("The workspace is {}".format(workspace))
        log("The cache workspace is {}".format(cache_workspace), INFO)
        log("The submission workspace is {}".format(submission_workspace), INFO)

        filepath_pkl = os.path.join(cache_workspace, "final_results.pkl")
        log("The filepath_pkl is {}".format(filepath_pkl), INFO)

        load_cache(filepath_pkl, is_hdb=True, others=(results, weight))

        final_submission_filename.append("-".join([stamp[:len(stamp)/3], str(weight)]))

    csv = transform_to_submission_format(results, 3)
    for size in [3]:
        filepath_output = "{}.{}.{}.csv".format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"), "_".join(final_submission_filename), size)
        save_submission(filepath_output, csv, size)

if __name__ == "__main__":
    facebook_weight()

#!/usr/bin/env python

import os
import sys
import datetime

import click
import pandas as pd

from utils import log, INFO
from facebook.facebook_learning import transform_to_submission_format, save_submission
from configuration import FacebookConfiguration

@click.command()
@click.option("--conf", required=True, help="Filepath of Configuration")
@click.option("--is-testing", is_flag=True, help="Turn on the testing mode")
def facebook_weight(conf, is_testing):
    configuration = FacebookConfiguration(conf)

    results = {}
    final_submission_filename = ["vote"]
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

        filepath_submission = submission_workspace + ".10.csv"

        log("start to read {}".format(filepath_submission), INFO)
        df = pd.read_csv(filepath_submission, dtype={"row_id": str, "place_id": str})
        for value in df.values:
            [row_id, place_ids] = value

            # No Place ID
            if isinstance(place_ids, float):
                continue

            results.setdefault(row_id, {})
            for place_id, vote in zip(place_ids.split(" "), [6, 5, 5, 5, 4, 4, 4, 3, 3, 3]):
                results[row_id].setdefault(place_id, 0)
                results[row_id][place_id] += vote

        final_submission_filename.append("-".join([stamp, str(weight)]))

    csv = transform_to_submission_format(results, 3)
    for size in [3]:
        filepath_output = "{}.{}.{}.csv".format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"), "_".join(final_submission_filename), size)
        save_submission(filepath_output, csv, size)

if __name__ == "__main__":
    facebook_weight()

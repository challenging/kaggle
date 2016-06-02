#!/usr/bin/env python

import os
import sys
import datetime

import click
import threading
import Queue

from utils import log, INFO
from facebook.facebook_learning import process, save_submission
from configuration import FacebookConfiguration

working_queue = Queue.Queue()

@click.command()
@click.option("--conf", required=True, help="Filepath of Configuration")
@click.option("--method-jobs", default=1, help="Number of thread for sections")
@click.option("--n-jobs", default=4, help="Number of thread of methods")
@click.option("--is-testing", is_flag=True, help="Testing Mode")
def facebook(conf, method_jobs, n_jobs, is_testing):
    global working_queue

    configuration = FacebookConfiguration(conf)

    for section in configuration.get_methods():
        working_queue.put(section)

    for idx in range(0, method_jobs):
        thread = threading.Thread(target=run, kwargs={"n_jobs": n_jobs, "is_testing": is_testing, "configuration": configuration})
        thread.setDaemon(True)
        thread.start()

    working_queue.join()

def run(n_jobs, is_testing, configuration):
    global working_queue

    while True:
        m = working_queue.get()

        workspace, cache_workspace, output_workspace = configuration.get_workspace(m)
        is_full = configuration.is_full()

        method, criteria, strategy, stamp, (window_size, batch_size, n_top), is_accuracy, is_exclude_outlier, is_normalization = configuration.get_method_detail(m)
        log("The method is {}, window_size is {}, batch_size is {}. n_top is {}. is_exclude_outlier is {}. is_accuracy is {}. is_normalization is {}".format(\
            method, window_size, batch_size, n_top, is_exclude_outlier, is_accuracy, is_normalization))

        setting = configuration.get_setting("{}-SETTING".format(m))
        setting_stamp = make_a_stamp(setting)

        filepath_train = os.path.join(workspace, "train.csv")
        filepath_test = os.path.join(workspace, "test.csv")

        normalization = "normalization_" if is_normalization else ""
        grid_size = criteria if isinstance(criteria, str) else "x".join(criteria)

        if method == "native":
            cache_workspace = "{}/{}criteria={}_windowsize={}_batchsize={}_isaccuracy={}_excludeoutlier={}_istesting={}/method={}.{}.{}/{}".format(\
                cache_workspace, normalization, grid_size, window_size, batch_size, is_accuracy, is_exclude_outlier, is_testing, method, stamp, n_top, setting_stamp)
            submission_workspace = "{}/{}criteria={}_windowsize={}_batchsize={}_isaccuracy={}_excludeoutlier={}_istesting={}/method={}.{}.{}/{}".format(\
                output_workspace, normalization, grid_size, window_size, batch_size, is_accuracy, is_exclude_outlier, is_testing, method, stamp, n_top, setting_stamp)
        else:
            cache_workspace = "{}/{}criteria={}_windowsize={}_batchsize={}_isaccuracy={}_excludeoutlier={}_istesting={}/method={}_strategy={}.{}.{}/{}".format(\
                cache_workspace, normalization, grid_size, window_size, batch_size, is_accuracy, is_exclude_outlier, is_testing, method, strategy, stamp, n_top, setting_stamp)
            submission_workspace = "{}/{}criteria={}_windowsize={}_batchsize={}_isaccuracy={}_excludeoutlier={}_istesting={}/method={}_strategy={}.{}.{}/{}".format(\
                output_workspace, normalization, grid_size, window_size, batch_size, is_accuracy, is_exclude_outlier, is_testing, method, strategy, stamp, n_top, setting_stamp)

        log("The workspace is {}".format(workspace))
        log("The cache workspace is {}".format(cache_workspace), INFO)
        log("The submission workspace is {}".format(submission_workspace), INFO)

        filepath_pkl = os.path.join(cache_workspace, "final_results.pkl")
        results = process((method, setting), (workspace, cache_workspace, submission_workspace), filepath_pkl, batch_size, criteria, strategy, is_accuracy, is_exclude_outlier, is_normalization, is_testing, n_top=n_top, n_jobs=max(1, n_jobs))

        if results:
            for size in [n_top, 3]:
                filepath_output = submission_workspace + ".{}.csv".format(size)
                save_submission(filepath_output, results, size, is_full=is_full)

        working_queue.task_done()

if __name__ == "__main__":
    facebook()

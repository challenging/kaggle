#!/usr/bin/env python

import os
import sys
import datetime

import click
import threading
import Queue

from utils import log, INFO
from facebook.facebook_utils import transform_to_submission_format
from facebook.facebook_learning import process
from configuration import FacebookConfiguration

working_queue = Queue.Queue()

@click.command()
@click.option("--conf", required=True, help="Filepath of Configuration")
@click.option("--method-jobs", default=1, help="Number of thread for sections")
@click.option("--n-jobs", default=4, help="Number of thread of methods")
@click.option("--is-beanstalk", is_flag=True, help="beanstalk mode")
@click.option("--is-testing", is_flag=True, help="Testing Mode")
def facebook(conf, method_jobs, n_jobs, is_beanstalk, is_testing):
    configuration = FacebookConfiguration(conf)

    global working_queue
    for section in configuration.get_methods():
        working_queue.put(section)

    for idx in range(0, method_jobs):
        thread = threading.Thread(target=run, kwargs={"n_jobs": n_jobs, "is_testing": is_testing, "is_beanstalk": is_beanstalk, "configuration": configuration})
        thread.setDaemon(True)
        thread.start()

    working_queue.join()

def run(n_jobs, is_testing, is_beanstalk, configuration):
    global working_queue

    while True:
        m = working_queue.get()

        workspace, cache_workspace, submission_workspace = configuration.get_workspace(m, is_testing)
        is_full = configuration.is_full()

        method, criteria, strategy, stamp, (window_size, batch_size, n_top), is_accuracy, is_exclude_outlier, is_normalization, dropout = configuration.get_method_detail(m)
        log("The method is {}, window_size is {}, batch_size is {}. n_top is {}. is_exclude_outlier is {}. is_accuracy is {}. is_normalization is {}. dropout is {}".format(\
            method, window_size, batch_size, n_top, is_exclude_outlier, is_accuracy, is_normalization, dropout))

        setting = configuration.get_setting("{}-SETTING".format(m))

        filepath_train = os.path.join(workspace, "train.csv")
        filepath_test = os.path.join(workspace, "test.csv")

        log("The workspace is {}".format(workspace))
        log("The cache workspace is {}".format(cache_workspace), INFO)
        log("The submission workspace is {}".format(submission_workspace), INFO)

        filepath_pkl = os.path.join(cache_workspace, "final_results.pkl")
        results = process((method, setting), (workspace, cache_workspace, submission_workspace),\
                            filepath_pkl, batch_size, criteria, strategy, is_accuracy, is_exclude_outlier, is_normalization, is_beanstalk, is_testing, dropout,\
                            n_top=n_top, n_jobs=max(1, n_jobs))

        if results:
            results = transform_to_submission_format(results, n_top)

            for size in [n_top, 3]:
                filepath_output = submission_workspace + ".{}.csv".format(size)
                save_submission(filepath_output, results, size, is_full=is_full)
        else:
            if not is_beanstalk:
                log("Get empty results", ERROR)

        working_queue.task_done()

if __name__ == "__main__":
    facebook()

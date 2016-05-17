#!/usr/bin/env python

import os
import sys
import datetime

import click
import threading
import Queue

BASEPATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/../lib".format(BASEPATH))

working_queue = Queue.Queue()

from utils import log, INFO
from facebook_learning import process, save_submission
from configuration import FacebookConfiguration

@click.command()
@click.option("--conf", required=True, help="Filepath of Configuration")
@click.option("--n-jobs", default=4, help="Number of thread")
@click.option("--is-testing", is_flag=True, help="Testing Mode")
def facebook(conf, n_jobs, is_testing):
    global working_queue

    configuration = FacebookConfiguration(conf)

    for section in configuration.get_methods():
        working_queue.put(section)

    for idx in range(0, 1):
        thread = threading.Thread(target=run, kwargs={"n_jobs": n_jobs, "is_testing": is_testing, "configuration": configuration})
        thread.setDaemon(True)
        thread.start()

    working_queue.join()

def run(n_jobs, is_testing, configuration):
    global working_queue

    while True:
        m = working_queue.get()

        workspace, cache_workspace, output_workspace = configuration.get_workspace(m)
        method, criteria, stamp, (window_size, batch_size, n_top), is_accuracy, is_exclude_outlier = configuration.get_method_detail(m)
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

        filepath_pkl = os.path.join(cache_workspace, "final_results.pkl")
        results = process(method, (workspace, cache_workspace, submission_workspace), filepath_pkl, batch_size, criteria, is_accuracy, is_exclude_outlier, is_testing, n_top=n_top, n_jobs=max(1, n_jobs/2))

        filepath_output = submission_workspace + ".csv.gz"
        for size in [n_top, 3]:
            filepath_output = submission_workspace + ".{}.csv.gz".format(size)
            save_submission(filepath_output, results, size)

        working_queue.task_done()

if __name__ == "__main__":
    facebook()

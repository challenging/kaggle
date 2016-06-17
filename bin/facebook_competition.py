#!/usr/bin/env python

import os
import sys

import click
import threading
import Queue

from utils import log, INFO
from facebook.facebook_utils import get_mongo_location
from facebook.facebook_learning import process
from configuration import FacebookConfiguration

working_queue = Queue.Queue()

@click.command()
@click.option("--conf", required=True, help="Filepath of Configuration")
@click.option("--is-testing", is_flag=True, help="Testing Mode")
def facebook(conf, is_testing):
    global working_queue

    configuration = FacebookConfiguration(conf)
    method_jobs = 1

    mapping_method = {}
    mapping_setting = {}
    for m in configuration.get_methods():
        workspace, cache_workspace, submission_workspace = configuration.get_workspace(m, is_testing)
        database, collection = get_mongo_location(cache_workspace)

        mapping_setting.setdefault(database, {})
        mapping_setting[database][collection] = configuration.get_setting("{}-SETTING".format(m))

        mapping_method[database] = m

    for database, collections in mapping_setting.items():
        working_queue.put((mapping_method[database], database, collections))

    for idx in range(0, method_jobs):
        thread = threading.Thread(target=run, kwargs={"is_testing": is_testing, "configuration": configuration})
        thread.setDaemon(True)
        thread.start()

    working_queue.join()

def run(is_testing, configuration):
    global working_queue

    while True:
        m, database, collections = working_queue.get()

        workspace, cache_workspace, submission_workspace = configuration.get_workspace(m, is_testing)
        log("The locations of mongo are {} - {}".format(database, collections), INFO)

        method, criteria, strategy, stamp, (window_size, batch_size, n_top), is_accuracy, is_exclude_outlier, is_normalization, dropout = configuration.get_method_detail(m)
        log("The method is {}, window_size is {}, batch_size is {}. n_top is {}. is_exclude_outlier is {}. is_accuracy is {}. is_normalization is {}. dropout is {}".format(\
            method, window_size, batch_size, n_top, is_exclude_outlier, is_accuracy, is_normalization, dropout))

        process(method, (workspace, database, collections),\
                criteria, strategy, is_accuracy, is_exclude_outlier, is_normalization, is_testing, dropout,\
                n_top=n_top)

        working_queue.task_done()

if __name__ == "__main__":
    facebook()

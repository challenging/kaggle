#!/usr/bin/env python

import os
import sys
import click
import datetime
import Queue

import pymongo

import numpy as np
import pandas as pd

from heapq import nlargest
from joblib import Parallel, delayed

from load import load_cache, import_hdb
from utils import create_folder, log, INFO
from facebook.facebook_score import get_full_queue, merge_files, NormalizeThread, WeightedThread, AggregatedThread
from facebook.facebook_utils import transform_to_submission_format, save_submission, get_mongo_location, _import_hdb
from facebook.facebook_utils import MONGODB_URL, MONGODB_INDEX, MONGODB_VALUE, MONGODB_SCORE, MONGODB_BATCH_SIZE, FULL_SET
from facebook.facebook_utils import MODE_VOTE, MODE_SIMPLE, MODE_WEIGHT, TEMP_FOLDER
from configuration import FacebookConfiguration

@click.command()
@click.option("--conf", required=True, help="filepath of Configuration")
@click.option("--mode", default="simple", help="ensemble mode")
@click.option("--n-jobs", default=1, help="number of thread")
@click.option("--n-top", default=3, help="number of predictions")
@click.option("--is-name", is_flag=True, help="just get mongo location")
@click.option("--is-score", is_flag=True, help="output the score")
@click.option("--is_local", is_flag=True, help="local mode")
@click.option("--is-import", is_flag=True, help="import HDB file into the MongoDB")
@click.option("--is-aggregate", is_flag=True, help="aggregation mode")
def facebook_ensemble(conf, mode, n_jobs, n_top, is_name, is_score, is_local, is_import, is_aggregate):
    configuration = FacebookConfiguration(conf)

    results = {}
    final_submission_filename = [mode]

    if is_name:
        for m in configuration.get_methods():
            workspace, cache_workspace, submission_workspace = configuration.get_workspace(m, False)
            log("{} --> The workspace is {}".format(m, workspace), INFO)

            get_mongo_location(cache_workspace)
    elif is_import:
        Parallel(n_jobs=6)(delayed(_import_hdb)(configuration, m) for m in configuration.get_methods())
    elif is_aggregate:
        mongo = pymongo.MongoClient(MONGODB_URL)

        for m in configuration.get_methods():
            workspace, cache_workspace, submission_workspace = configuration.get_workspace(m, False)
            database, collection = get_mongo_location(cache_workspace)

            count = mongo[database][collection].count()
            if count < FULL_SET[1]+500000:
                log("Skip the aggregation for {}-{}".format(database, collection), INFO)
                continue

            new_collection = "{}_aggregation".format(collection)
            # create index for new collection
            mongo[database][new_collection].create_index(MONGODB_INDEX)
            log("Create index for new collection", INFO)

            queue = get_full_queue(100000)
            for i in range(0, n_jobs):
                thread = AggregatedThread(kwargs={"queue": queue, "database": database, "collection": collection})
                thread.setDaemon(True)
                thread.start()
            queue.join()

            log("Finish aggregating the values for {}-{}".format(database, collection), INFO)

            mongo[database][collection].drop()
            log("Drop the {}".format(collection), INFO)

            mongo[database][new_collection].rename(collection)
            log("Rename {} to {}".format(new_collection ,collection), INFO)

        mongo.close()
    else:
        locations = []

        if mode == MODE_SIMPLE:
            for m in configuration.get_methods():
                workspace, cache_workspace, submission_workspace = configuration.get_workspace(m, False)
                log("The workspace is {}".format(workspace), INFO)

                database, collection = get_mongo_location(cache_workspace)
                weights = [1]
                dropout = configuration.get_value(m, "dropout")
                locations = [(database, collection, weights, True if dropout and dropout.isdigit() else False)]

                queue = get_full_queue(100000)

                filepath_prefix = os.path.join(mode, os.path.basename(conf), m.lower())
                if is_score:
                    filepath_prefix = TEMP_FOLDER

                create_folder("{}/1.txt".format(filepath_prefix))

                for idx in range(0, n_jobs):
                    thread = WeightedThread(kwargs={"mode": mode, "queue": queue, "locations": locations, "filepath_prefix": filepath_prefix, "is_score": is_score, "ntop": n_top})
                    thread.setDaemon(True)
                    thread.start()

                queue.join()

                if not is_score:
                    for idx in range(0, len(weights)):
                        merge_files(filepath_prefix, idx, final_submission_filename)
        elif mode == MODE_WEIGHT:
            total_dropout, total_o_dropout = 0, 0
            for m in configuration.get_methods():
                workspace, cache_workspace, submission_workspace = configuration.get_workspace(m, False)
                log("The workspace is {}".format(workspace), INFO)

                database, collection = get_mongo_location(cache_workspace)
                weights = configuration.get_weight(m)

                dropout = configuration.get_value(m, "dropout")
                if dropout and dropout.isdigit():
                    total_dropout += 1
                else:
                    total_o_dropout += 1

                locations.append([database, collection, weights, True if dropout and dropout.isdigit() else False])

            adjust = None
            if total_dropout == 0:
                adjust = 1
            else:
                adjust = float(total_o_dropout)/total_dropout
                if adjust == 0:
                    adjust = 1

            for location in locations:
                location[-1] = adjust if location[-1] else 1
                log(location)

            queue = Queue.Queue()
            for database, collection, _, _ in locations:
                queue.put((database, collection))

            for idx in range(0, n_jobs):
                thread = NormalizeThread(kwargs={"queue": queue})
                thread.setDaemon(True)
                thread.start()

            queue.join()
            log("Finish getting the min and max values", INFO)

            queue = get_full_queue(100000)

            filepath_prefix = os.path.join(mode, os.path.basename(conf))

            create_folder("{}/1.txt".format(filepath_prefix))
            for idx in range(0, n_jobs):
                thread = WeightedThread(kwargs={"mode": mode, "queue": queue, "locations": locations, "filepath_prefix": filepath_prefix, "is_score": is_score, "ntop": n_top})
                thread.setDaemon(True)
                thread.start()

            queue.join()

            # merge file
            for idx in range(0, len(weights)):
                merge_files(filepath_prefix, idx, final_submission_filename)

if __name__ == "__main__":
    facebook_ensemble()

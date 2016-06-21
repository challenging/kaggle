#!/usr/bin/env python

import os
import sys
import click
import datetime

import pymongo
import Queue

import numpy as np
import pandas as pd

from heapq import nlargest
from joblib import Parallel, delayed

from load import load_cache, import_hdb
from utils import create_folder, log, INFO
from facebook.facebook_score import get_full_queue, merge_files, NormalizeThread, WeightedThread
from facebook.facebook_utils import transform_to_submission_format, save_submission, get_mongo_location, _import_hdb
from facebook.facebook_utils import MONGODB_URL, MONGODB_INDEX, MONGODB_VALUE, MONGODB_SCORE, MONGODB_BATCH_SIZE, FULL_SET
from configuration import FacebookConfiguration

@click.command()
@click.option("--conf", required=True, help="filepath of Configuration")
@click.option("--mode", default="vote", help="ensemble mode")
@click.option("--n-jobs", default=4, help="number of thread")
@click.option("--is-name", is_flag=True, help="just get mongo location")
@click.option("--is-import", is_flag=True, help="import HDB file into the MongoDB")
@click.option("--is-beanstalk", is_flag=True, help="beanstalk mode")
def facebook_ensemble(conf, mode, n_jobs, is_name, is_import, is_beanstalk):
    configuration = FacebookConfiguration(conf)

    results = {}
    final_submission_filename = [mode]

    def get_filepath_prefix(mode, conf, m):
        folder = "{}/{}/{}".format(mode, os.path.basename(conf), m.lower())
        create_folder(folder + "/1.txt")

        return folder

    if is_name:
        for m in configuration.get_methods():
            workspace, cache_workspace, submission_workspace = configuration.get_workspace(m, False)
            log("The workspace is {}".format(workspace), INFO)

            get_mongo_location(cache_workspace)
    elif is_beanstalk:
        locations = []

        if mode == "simple":
            for m in configuration.get_methods():
                workspace, cache_workspace, submission_workspace = configuration.get_workspace(m, False)
                log("The workspace is {}".format(workspace), INFO)

                database, collection = get_mongo_location(cache_workspace)
                weight = 1
                locations = [(database, collection, weight)]

                queue = get_full_queue()

                filepath_prefix = get_filepath_prefix(mode, conf, m)
                for idx in range(0, n_jobs):
                    thread = WeightedThread(kwargs={"mode": mode, "queue": queue, "locations": locations, "filepath_prefix": filepath_prefix})
                    thread.setDaemon(True)
                    thread.start()

                queue.join()

                merge_files(filepath_prefix, conf, m, final_submission_filename)
        else:
            total_dropout, total_o_dropout = 0, 0
            for m in configuration.get_methods():
                workspace, cache_workspace, submission_workspace = configuration.get_workspace(m, False)
                log("The workspace is {}".format(workspace), INFO)

                database, collection = get_mongo_location(cache_workspace)
                weight = configuration.get_weight(m)

                dropout = configuration.get_value(m, "dropout")
                if dropout and dropout.isdigit():
                    total_dropout += 1
                else:
                    total_o_dropout += 1

                locations.append([database, collection, weight, True if dropout and dropout.isdigit() else False])

            adjust = float(total_o_dropout)/total_dropout
            for location in locations:
                location[-1] = adjust if location[-1] else 1/adjust

            queue = Queue.Queue()
            for database, collection, _, _ in locations:
                queue.put((database, collection))

            for idx in range(0, n_jobs):
                thread = NormalizeThread(kwargs={"queue": queue})
                thread.setDaemon(True)
                thread.start()
            queue.join()
            log("Finish getting the min and max values", INFO)

            queue = get_full_queue()

            filepath_prefix = get_filepath_prefix(mode, conf, m)
            for idx in range(0, n_jobs):
                thread = WeightedThread(kwargs={"mode": mode, "queue": queue, "locations": locations, "filepath_prefix": filepath_prefix})
                thread.setDaemon(True)
                thread.start()

            queue.join()

            # merge file
            merge_files(filepath_prefix, conf, m, final_submission_filename)
    elif is_import:
        Parallel(n_jobs=6)(delayed(_import_hdb)(configuration, m) for m in configuration.get_methods())
    else:
        for m in configuration.get_methods():
            workspace, cache_workspace, submission_workspace = configuration.get_workspace(m, False)
            log("The workspace is {}".format(workspace), INFO)

            weight = configuration.get_weight(m)

            if mode == "weight":
                filepath_pkl = os.path.join(cache_workspace, "final_results.pkl")
                log("The filepath_pkl is {}".format(filepath_pkl), INFO)
                load_cache(filepath_pkl, is_hdb=True, others=(results, weight))
                final_submission_filename.append("-".join([m, str(weight)]))
            elif mode == "vote":
                filepath_submission = submission_workspace + ".10.csv"
                log("start to read {}".format(filepath_submission), INFO)

                df = pd.read_csv(filepath_submission, dtype={"row_id": str, "place_id": str})

                for value in df.values:
                    [row_id, place_ids] = value

                    # No Place ID
                    if isinstance(place_ids, float):
                        continue

                    results.setdefault(row_id, {})
                    n_end = 6
                    for place_id, vote in zip(place_ids.split(" "), [(n_end-idx) for idx in range(0, n_end)]):
                        results[row_id].setdefault(place_id, 0)
                        results[row_id][place_id] += vote
            else:
                raise NotImplementError

        if results:
            size = 3
            csv = transform_to_submission_format(results, size)
            filepath_output = "{}.{}.{}.csv".format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"), "_".join(final_submission_filename), size)
            save_submission(filepath_output, csv, size, is_full=FULL_SET)

if __name__ == "__main__":
    facebook_ensemble()

#!/usr/bin/env python

import os
import sys
import click
import datetime

import pymongo
import threading
import Queue

import pandas as pd

from load import load_cache
from utils import log, INFO
from facebook.facebook_beanstalk import get_mongo_location
from facebook.facebook_utils import transform_to_submission_format, save_submission
from facebook.facebook_utils import MONGODB_URL, MONGODB_INDEX
from configuration import FacebookConfiguration

def run(queue, locations, filepath_prefix, batch_size=5000):
    mongo = pymongo.MongoClient(MONGODB_URL)

    while True:
        idx_min, idx_max = queue.get()

        results = {}
        for database, collection, weight in locations:
            for record in mongo[database][collection].find({MONGODB_INDEX: {"$gte": idx_min, "$lt": idx_max}}).batch_size(batch_size):
                row_id = record["row_id"]
                results.setdefault(row_id, {})

                for place_id in record["place_ids"]:
                    results[row_id][place_id["place_id"]] = place_id["score"]

        csv = transform_to_submission_format(results, 3)
        filepath_output = "{}.{}.3.csv".format(filepath_prefix, threading.current_thread().id)
        save_submission(filepath_output, csv, size)

        queue.task_done()

    mongo.close()

@click.command()
@click.option("--conf", required=True, help="filepath of Configuration")
@click.option("--mode", default="vote", help="ensemble mode")
@click.option("--n-jobs", default=4, help="number of thread")
@click.option("--is-testing", is_flag=True, help="testing mode")
@click.option("--is-beanstalk", is_flag=True, help="beanstalk mode")
def facebook_ensemble(conf, mode, n_jobs, is_testing, is_beanstalk):
    configuration = FacebookConfiguration(conf)

    results = {}
    final_submission_filename = ["weight"]

    if is_beanstalk:
        locations = []

        for m in configuration.get_methods():
            workspace, cache_workspace, submission_workspace = configuration.get_workspace(m, False)
            log("The workspace is {}".format(workspace), INFO)

            database, collection = get_mongo_locations(cache_workspace)
            weight = configuration.get_weight(m)

            locations.append((database, collection ,weight))

        max_num = 8607230
        batch_num = 20000
        batch_idx = max_num/batch_num

        queue = Queue.Queue()
        for idx in xrange(0, batch_idx+1):
            queue.put((idx*batch_num, max(max_num, (idx+1)*batch_num)))

        filepath_prefix = "{}".format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
        for idx in range(0, 4):
            thread = threading.Thread(target=run, kwargs={"queue": queue, "locations": locations, "filepath_prefix": filepath_prefix})
            thread.setDaemon(True)
            thread.start()

        queue.join()
    else:
        for m in configuration.get_methods():
            workspace, cache_workspace, submission_workspace = configuration.get_workspace(m, False)
            log("The workspace is {}".format(workspace), INFO)

            weight = configuration.get_weight(m)

            if mode == "weight":
                filepath_pkl = os.path.join(cache_workspace, "final_results.pkl")
                log("The filepath_pkl is {}".format(filepath_pkl), INFO)
                load_cache(filepath_pkl, is_hdb=True, others=(results, weight))
                final_submission_filename.append("-".join([stamp[:len(stamp)/3], str(weight)]))
            elif mode == "vote":
                filepath_submission = submission_workspace + ".10.csv"
                log("start to read {}".format(filepath_submission), INFO)

                df = pd.read_csv(filepath_submission, dtype={"row_id": str, "place_id": str})

                count = 0
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

                    count += 1
                    if is_testing and count > 10000:
                        break

        csv = transform_to_submission_format(results, 3)
        filepath_output = "{}.{}.3.csv".format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"), "_".join(final_submission_filename))
        save_submission(filepath_output, csv, size, is_full=True)

if __name__ == "__main__":
    facebook_ensemble()

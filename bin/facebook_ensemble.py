#!/usr/bin/env python

import os
import sys
import click
import time
import datetime
import glob
import subprocess

import pymongo
import threading
import Queue

import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from load import load_cache, import_hdb
from utils import create_folder, log, INFO
from facebook.facebook_beanstalk import get_mongo_location
from facebook.facebook_utils import transform_to_submission_format, save_submission
from facebook.facebook_utils import MONGODB_URL, MONGODB_INDEX, MONGODB_VALUE, MONGODB_SCORE, MONGODB_BATCH_SIZE
from configuration import FacebookConfiguration

def normalize(queue):
    mongo = pymongo.MongoClient(MONGODB_URL)
    mm_database, mm_collection = "facebook_checkin_competition", "min_max"

    while True:
        database, collection = queue.get()
        log("Start to search the values for {} of {}".format(collection ,database), INFO)

        rmin, rmax = np.inf, -np.inf

        # Check the existing of min, max values for collections
        for r in mongo[mm_database][mm_collection].find({"database": database, "collection": collection}):
            rmin, rmax = r["min"], r["max"]

        # Not found the min/max records
        if rmin == np.inf:
            row_n = 0
            xx, x, n = 0.0, 0.0, 0
            for record in mongo[database][collection].find({}, {MONGODB_VALUE: 1}):
                for info in record[MONGODB_VALUE]:
                    score = info[MONGODB_SCORE]

                    xx += score**2
                    x += score

                    v_min = min(rmin, score)
                    v_max = max(rmax, score)

                    if v_min != rmin:
                        rmin = v_min

                    if v_max != rmax:
                        rmax = v_max

                    n += 1

                if row_n % 50000 == 0:
                    log("{}/{}/{}/{}/{} in {} of {}".format(n, x, xx, rmin, rmax, collection, database), INFO)
                row_n += 1

            avg = x/n
            std = np.sqrt(xx/n - avg**2)

            mongo[mm_database][mm_collection].insert({"database": database, "collection": collection, "std": std, "avg": avg, "n": n, "min": rmin, "max": rmax})

        log("Get {}/{} from {} of {}".format(rmin, rmax, collection, database), INFO)

        queue.task_done()

    mongo.close()

def weighted(score, weight):
    return score*1000**weight

def run(mode, queue, locations, filepath_prefix, batch_size=5000):
    mongo = pymongo.MongoClient(MONGODB_URL)

    def scoring(results, pre_row_id, pre_place_ids, avg, std, min_std, weight, eps, mode):
        results.setdefault(pre_row_id, {})
        size = len(pre_place_ids)

        if mode == "weight":
            for place_ids in pre_place_ids:
                for place_id in place_ids:
                    results[pre_row_id].setdefault(place_id["place_id"], 0)

                    score = (place_id["score"]-avg)/std+min_std+eps
                    results[pre_row_id][place_id["place_id"]] += weighted(score, weight)/size
        elif mode == "vote":
            score = {0: 10,
                     1: 9,
                     2: 9,
                     3: 8,
                     4: 8,
                     5: 7,
                     6: 7,
                     7: 6,
                     8: 6,
                     9: 5}

            for place_ids in pre_place_ids:
                for c, place_id in enumerate(sorted(place_ids, key=lambda x: x.values()[1], reverse=True)):
                    results[pre_row_id].setdefault(place_id["place_id"], 0)

                    results[pre_row_id][place_id["place_id"]] += score[c]/size


    eps = 0.0001
    mm_database, mm_collection = "facebook_checkin_competition", "min_max"
    while True:
        idx_min, idx_max = queue.get()

        results = {}
        for database, collection, weight in locations:
            avg, std, min_std = 0, 0, 0
            for r in mongo[mm_database][mm_collection].find({"database": database, "collection": collection}):
                avg = r["avg"]
                std = r["std"]
                min_std = (r["min"] - r["avg"])/r["std"]*-1+0.0001

            pre_row_id = None
            pre_place_ids = []

            timestamp_start = time.time()
            for record in mongo[database][collection].find({MONGODB_INDEX: {"$gte": idx_min, "$lt": idx_max}}).sort([(MONGODB_INDEX, pymongo.ASCENDING)]).batch_size(batch_size):
                row_id = record[MONGODB_INDEX]

                if pre_row_id != None and pre_row_id != row_id:
                    scoring(results, pre_row_id, pre_place_ids, avg, std, min_std, weight, eps, mode)
                    pre_place_ids = []

                pre_row_id = row_id
                pre_place_ids.append(record["place_ids"])

            scoring(results, pre_row_id, pre_place_ids, avg, std, min_std, weight, eps, mode)

            timestamp_end = time.time()
            log("Cost {:4f} secends to finish this job({} - {}) from {} of {} with {}".format((timestamp_end-timestamp_start), idx_min, idx_max, collection, database, weight), INFO)

        size = 3
        csv = transform_to_submission_format(results, size)
        filepath_output = "{}/{}.{}.{}.csv".format(filepath_prefix, threading.current_thread().ident, size, idx_min)
        save_submission(filepath_output, csv, size, is_full=[idx_min, idx_max])

        queue.task_done()

    mongo.close()

def _import_hdb(configuration, m):
    mongo = pymongo.MongoClient(MONGODB_URL)

    workspace, cache_workspace, submission_workspace = configuration.get_workspace(m, False)
    log("The workspace is {}".format(workspace), INFO)

    filepath_pkl = os.path.join(cache_workspace, "final_results.pkl")
    log("The filepath_pkl is {}".format(filepath_pkl), INFO)

    database, collection = get_mongo_location(cache_workspace)
    import_hdb(filepath_pkl, mongo[database][collection])

    mongo.close()

@click.command()
@click.option("--conf", required=True, help="filepath of Configuration")
@click.option("--mode", default="vote", help="ensemble mode")
@click.option("--n-jobs", default=4, help="number of thread")
@click.option("--is-import", is_flag=True, help="import HDB file into the MongoDB")
@click.option("--is-beanstalk", is_flag=True, help="beanstalk mode")
def facebook_ensemble(conf, mode, n_jobs, is_import, is_beanstalk):
    configuration = FacebookConfiguration(conf)

    results = {}
    final_submission_filename = [mode]

    if is_beanstalk:
        locations = []

        for m in configuration.get_methods():
            workspace, cache_workspace, submission_workspace = configuration.get_workspace(m, False)
            log("The workspace is {}".format(workspace), INFO)

            database, collection = get_mongo_location(cache_workspace)
            weight = configuration.get_weight(m)

            locations.append((database, collection, weight))

        queue = Queue.Queue()
        for database, collection, _ in locations:
            queue.put((database, collection))

        for idx in range(0, n_jobs):
            thread = threading.Thread(target=normalize, kwargs={"queue": queue})
            thread.setDaemon(True)
            thread.start()
        queue.join()
        log("Finish getting the min and max values", INFO)

        max_num = 8607230
        batch_num = 100000
        batch_idx = max_num/batch_num

        queue = Queue.Queue()
        for idx in xrange(0, batch_idx+1):
            queue.put((idx*batch_num, min(max_num, (idx+1)*batch_num)))

        filepath_prefix = "{}/{}".format(mode, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
        create_folder(filepath_prefix + "/1.txt")
        for idx in range(0, n_jobs):
            thread = threading.Thread(target=run, kwargs={"mode": mode, "queue": queue, "locations": locations, "filepath_prefix": filepath_prefix})
            thread.setDaemon(True)
            thread.start()

        queue.join()

        # merge file
        filepath_final = "{}/final.csv.gz".format(filepath_prefix)
        filepath_final = "{}/{}.{}.{}.3.csv.gz".format(filepath_prefix, os.path.basename(conf).replace(".cfg", ""), m.lower(), "_".join(final_submission_filename))
        rc = subprocess.call("echo 'row_id,place_id' | gzip -9 > {}".format(filepath_final), shell=True)

        for f in glob.iglob("{}/*.csv".format(filepath_prefix)):
            rc = subprocess.call("tail -n +2 {} | gzip -9 >> {}".format(f, filepath_final), shell=True)
            log("Append {} to the end of {}".format(f, filepath_final), INFO)
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
            elif mode == "simple":
                size = 3

                filepath_submission = submission_workspace + ".{}.csv".format(size)
                create_folder(filepath_submission)
                log("The submission filepath is {}".format(filepath_submission), INFO)

                filepath_output = "{}.{}.{}.{}.{}.csv".format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"), os.path.basename(conf).replace(".cfg", ""), m.lower(), "_".join(final_submission_filename), size)

                mongo = pymongo.MongoClient(MONGODB_URL)

                pre_row_id = None
                database, collection = get_mongo_location(cache_workspace)

                def score(n_top, ranking):
                    line = []

                    for place_id, most_popular in nlargest(n_top, sorted(ranking.items()), key=lambda (k, v): v):
                        line.append(str(place_id))

                    return " ".join(line)

                with open(filepath_output, "wb") as OUTPUT:
                    pool = {}
                    OUTPUT.write("row_id,place_id\n")

                    for count, record in enumerate(mongo[database][collection].find({}).sort([(MONGODB_INDEX, pymongo.ASCENDING)]).batch_size(MONGODB_BATCH_SIZE)):
                        row_id = str(record[MONGODB_INDEX])

                        if pre_row_id != None and pre_row_id != row_id:
                            OUTPUT.write("{},{}\n".format(row_id, score(size, pool)))

                            pool = {}

                        pool.setdefault(row_id, {})
                        for info in record[MONGODB_VALUE]:
                            place_id, score = info[MONGODB_VALUE[:-1]], info[MONGODB_SCORE]

                            pool[row_id].setdefault(place_id, 0)
                            pool[row_id][place_id] += score

                        if count % 500000 == 0 and count > 0:
                            log("The progress is {}".format(count), INFO)

                    OUTPUT.write("{},{}\n".format(pre_row_id, score(size, pool)))

                mongo.close()

        if results:
            size = 3
            csv = transform_to_submission_format(results, size)
            filepath_output = "{}.{}.{}.csv".format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"), "_".join(final_submission_filename), size)
            save_submission(filepath_output, csv, size, is_full=FULL_SET)

if __name__ == "__main__":
    facebook_ensemble()

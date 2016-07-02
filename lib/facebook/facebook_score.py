#!/usr/bin/env python

import os
import copy
import glob
import time
import subprocess

import pymongo
import threading
import Queue

import numpy as np

from utils import create_folder, log, INFO, WARN
from facebook.facebook_utils import transform_to_submission_format, save_submission, get_mongo_location
from facebook.facebook_utils import MONGODB_URL, MONGODB_INDEX, MONGODB_VALUE, MONGODB_SCORE, MONGODB_BATCH_SIZE, FULL_SET
from facebook.facebook_utils import MODE_SIMPLE, MODE_WEIGHT

def get_full_queue(batch_num=200000):
    max_num = FULL_SET[1]
    batch_idx = max_num/batch_num

    queue = Queue.Queue()
    for idx in xrange(0, batch_idx+1):
        queue.put((idx*batch_num, min(max_num, (idx+1)*batch_num)))

    return queue

def merge_files(filepath_prefix, idx, filename_submission):
    filepath_final = "{}/{}.{}.csv.gz".format(filepath_prefix, idx, "_".join(filename_submission))
    rc = subprocess.call("echo 'row_id,place_id' | gzip -9 > {}".format(filepath_final), shell=True)

    log("{}/{}.*.csv".format(filepath_prefix, idx), INFO)
    for f in glob.iglob("{}/{}.*.csv".format(filepath_prefix, idx)):
        rc = subprocess.call("tail -n +2 {} | gzip -9 >> {}".format(f, filepath_final), shell=True)
        log("Append {} to the end of {}".format(f, filepath_final), INFO)

class NormalizeThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        threading.Thread.__init__(self, group=group, target=target, name=name, verbose=verbose)

        self.args = args

        for key, value in kwargs.items():
            setattr(self, key, value)

    def run(self):
        mongo = pymongo.MongoClient(MONGODB_URL)
        mm_database, mm_collection = "facebook_checkin_competition", "min_max"

        while True:
            database, collection = self.queue.get()
            log("Start to search the values for {} of {}".format(collection ,database), INFO)

            rmin, rmax = np.inf, -np.inf

            # Check the existing of min, max values for collections
            for r in mongo[mm_database][mm_collection].find({"database": database, "collection": collection}):
                rmin, rmax = r["min"], r["max"]

            # Not found the min/max records
            if rmin == np.inf:
                row_n = 0
                xx, x, n = 0.0, 0.0, 0
                for record in mongo[database][collection].find({}, {MONGODB_VALUE: 1}).batch_size(MONGODB_BATCH_SIZE):
                    for info in record[MONGODB_VALUE]:
                        scores = info[MONGODB_SCORE]

                        if isinstance(scores, float):
                            scores = [scores]

                        for score in scores:
                            xx += score**2
                            x += score

                            v_min = min(rmin, score)
                            v_max = max(rmax, score)

                            if v_min != rmin:
                                rmin = v_min

                            if v_max != rmax:
                                rmax = v_max

                            n += 1

                    if row_n % 200000 == 0:
                        log("{}/{}/{}/{} in {} of {}".format(row_n, n, rmin, rmax, collection, database), INFO)
                    row_n += 1

                avg = x/n
                std = np.sqrt(xx/n - avg**2)

                mongo[mm_database][mm_collection].insert({"database": database, "collection": collection, "std": std, "avg": avg, "n": n, "min": rmin, "max": rmax})

            log("Get {}/{} from {} of {}".format(rmin, rmax, collection, database), INFO)

            self.queue.task_done()

        mongo.close()

class AggregatedThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        threading.Thread.__init__(self, group=group, target=target, name=name, verbose=verbose)

        self.args = args

        for key, value in kwargs.items():
            setattr(self, key, value)

    def run(self):
        mongo = pymongo.MongoClient(MONGODB_URL)
        new_collection = self.collection + "_aggregation"

        def get_place_ids(place_ids):
            r = {}

            for rr in place_ids:
                place_id, score = rr["place_id"], rr[MONGODB_SCORE]

                r.setdefault(place_id, [])
                r[place_id].append(score)

            return r

        idx_min, idx_max = None, None
        while True:
            if idx_min == None or idx_max == None:
                idx_min, idx_max = self.queue.get()

            try:
                pre_row_id = None
                pre_place_ids, pre_grids = [], []
                pool = []

                count = mongo[self.database][new_collection].count({MONGODB_INDEX: {"$gte": idx_min, "$lt": idx_max}})

                if count == (idx_max-idx_min):
                    log("Skipping {} to {}".format(idx_min, idx_max), INFO)

                    idx_min, idx_max = None, None
                    self.queue.task_done()

                    continue
                elif count > 0:
                    log(mongo[self.database][new_collection].remove({MONGODB_INDEX: {"$gte": idx_min, "$lt": idx_max}}), INFO)

                log("Start to aggregate the values for {} of {}".format(idx_min, idx_max), INFO)
                timestamp_start = time.time()
                for record in mongo[self.database][self.collection].find({MONGODB_INDEX: {"$gte": idx_min, "$lt": idx_max}}).sort([(MONGODB_INDEX, pymongo.ASCENDING)]).batch_size(MONGODB_BATCH_SIZE):
                    row_id = record[MONGODB_INDEX]

                    if pre_row_id != None and pre_row_id != row_id:
                        r = {MONGODB_INDEX: pre_row_id,
                             MONGODB_VALUE: get_place_ids(pre_place_ids)}
                        pool.append(r)

                        pre_place_ids = []

                    pre_row_id = row_id
                    pre_place_ids.extend(record["place_ids"])

                r = {MONGODB_INDEX: pre_row_id,
                     MONGODB_VALUE: get_place_ids(pre_place_ids)}
                pool.append(r)

                records = []
                for r in pool:
                    row_id = r[MONGODB_INDEX]

                    values = []
                    for place_id, scores in r[MONGODB_VALUE].items():
                        values.append({"place_id": place_id, MONGODB_SCORE: scores})

                    records.append({MONGODB_INDEX: row_id, MONGODB_VALUE: values})

                timestamp_end = time.time()
                log("Cost {:4f} secends to query records".format(timestamp_end-timestamp_start), INFO)

                timestamp_start = time.time()
                mongo[self.database][new_collection].insert_many(records)
                timestamp_end = time.time()
                log("Cost {:4f} secends to insert records".format(timestamp_end-timestamp_start), INFO)
            except pymongo.errors.CursorNotFound as e:
                log(e)
                time.sleep(60)

                continue
            else:
                idx_min, idx_max = None, None

            self.queue.task_done()

        mongo.close()

class WeightedThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        threading.Thread.__init__(self, group=group, target=target, name=name, verbose=verbose)

        self.args = args

        for key, value in kwargs.items():
            setattr(self, key, value)

    def run(self):
        eps = 0.00000001
        batch_size = 10000
        mongo = pymongo.MongoClient(MONGODB_URL)

        def scoring(results, row_id, pre_place_ids, avg, std, min_std, adjust, weights):
            for result in results:
                result.setdefault(row_id, {})

            if self.mode == MODE_SIMPLE:
                for place_ids in pre_place_ids:
                    for place_id in place_ids:
                        results[0][row_id].setdefault(place_id["place_id"], 0)

                        for score in place_id["score"]:
                            results[0][row_id][place_id["place_id"]] += score
            elif self.mode == MODE_WEIGHT:
                for idx, result in enumerate(results):
                    if weights[idx]:
                        for place_ids in pre_place_ids:
                            size = 1
                            for p in place_ids:
                                scores = p[MONGODB_SCORE]
                                if isinstance(scores, list):
                                    size = max(size, len(scores))

                            for p in place_ids:
                                place_id, scores = p["place_id"], p[MONGODB_SCORE]

                                result[row_id].setdefault(place_id, 0)

                                if isinstance(scores, float):
                                    scores = [scores]

                                for score in scores:
                                    score = ((score-avg)/(std+eps)+min_std)*weights[idx]/size*adjust
                                    result[row_id][place_id] += score

                                    #if row_id in [50088, 89377, 440138]:
                                    #    log((idx, weights, row_id, place_id, size, score, result[row_id][place_id]), INFO)
            else:
                raise NotImplementError

        size = 3
        idx_min, idx_max = None, None
        mm_database, mm_collection = "facebook_checkin_competition", "min_max"
        while True:
            if idx_min == None or idx_max == None:
                idx_min, idx_max = self.queue.get()

            try:
                first_weights = self.locations[0][2]
                results = [{} for idx in range(0, len(first_weights))] # Get the weights of the first element

                locations = copy.deepcopy(self.locations)
                for database, collection, weights, adjust in locations:
                    for idx in range(0, len(first_weights)):
                        filepath_output = "{}/{}.{}.csv".format(self.filepath_prefix, idx, idx_min)
                        if os.path.exists(filepath_output):
                            log("Skipping the {}({}) for {}-{}".format(idx, weights[idx], database, collection), INFO)

                            weights[idx] = None

                for idx, (database, collection, weights, adjust) in enumerate(locations):
                    all_done = True
                    for weight in weights:
                        all_done &= (weight == None)

                    if all_done:
                        continue

                    avg, std, min_std = 0, 0, 0
                    for r in mongo[mm_database][mm_collection].find({"database": database, "collection": collection}):
                        avg = r["avg"]
                        std = r["std"]
                        min_std = (r["min"] - r["avg"])/(r["std"]+eps)*-1

                    pre_row_id = None
                    pre_place_ids = []

                    timestamp_start = time.time()
                    for record in mongo[database][collection].find({MONGODB_INDEX: {"$gte": idx_min, "$lt": idx_max}}).sort([(MONGODB_INDEX, pymongo.ASCENDING)]).batch_size(MONGODB_BATCH_SIZE):
                        row_id = record[MONGODB_INDEX]

                        if pre_row_id != None and pre_row_id != row_id:
                            scoring(results, pre_row_id, pre_place_ids, avg, std, min_std, adjust, weights)
                            pre_place_ids = []

                        pre_row_id = row_id
                        pre_place_ids.append(record["place_ids"])

                    if pre_row_id:
                        scoring(results, pre_row_id, pre_place_ids, avg, std, min_std, adjust, weights)

                    timestamp_end = time.time()
                    log("Cost {:4f} secends to finish this job({} - {}) from {} - {} with {}".format((timestamp_end-timestamp_start), idx_min, idx_max, database, collection, weights), INFO)

                for idx, result in enumerate(results):
                    filepath_output = "{}/{}.{}.csv".format(self.filepath_prefix, idx, idx_min)
                    if not os.path.exists(filepath_output):
                        csv = transform_to_submission_format(result, size)
                        save_submission(filepath_output, csv, size, is_full=[idx_min, idx_max])
            except pymongo.errors.CursorNotFound as e:
                log(e, WARN)
                time.sleep(60)

                continue
            else:
                idx_min, idx_max = None, None

            self.queue.task_done()

        mongo.close()

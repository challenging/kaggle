#!/usr/bin/env python

import os
import sys
import math
import time

import threading
import Queue

import pandas as pd
import numpy as np

from scandir import scandir
from operator import itemgetter
from heapq import nlargest
from scipy import stats
from memory_profiler import profile

from facebook_strategy import StrategyEngine
from utils import log, DEBUG, INFO, WARN, ERROR
from utils import create_folder, make_a_stamp
from load import save_cache, load_cache

class BaseCalculatorThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        threading.Thread.__init__(self, group=group, target=target, name=name, verbose=verbose)

        self.args = args

        for key, value in kwargs.items():
            setattr(self, key, value)

        if not hasattr(self, "weights"):
            self.weights = 1

    def update_results(self, results, is_adjust=False):
        for test_id, clusters in results.items():
            self.results.setdefault(test_id, {})

            for place_id, score in clusters.items():
                self.results[test_id].setdefault(place_id, 0)
                self.results[test_id][place_id] += score*self.weights

    def run(self):
        while True:
            timestamp_start = time.time()

            test_ids, test_xs = self.queue.get()

            top = self.process(test_ids, test_xs)
            self.update_results(top, True)

            self.queue.task_done()

            timestamp_end = time.time()
            log("Cost {:4f} secends to finish this batch job({} - {}, {}) getting TOP-{} clusters. The remaining size of queue is {}".format(\
                timestamp_end-timestamp_start, test_ids[0], test_ids[-1], len(top), self.n_top, self.queue.qsize()), INFO)

    def process(self, test_ids, test_xs):
        raise NotImplementError

class MostPopularThread(BaseCalculatorThread):
    def process(self, test_ids, test_xs):
        stamp = make_a_stamp(str(test_ids) + str(test_xs))
        filepath = os.path.join(self.cache_workspace, "{}.pkl".format(stamp))

        top = load_cache(filepath)
        if not top or self.is_testing:
            top = {}

            count_missing = 0
            for test_id, test_x in zip(test_ids, test_xs):
                top.setdefault(test_id, {})

                key = (self.transformer(test_x[0], self.range_x[0], self.range_x[1], self.range_x[2]), self.transformer(test_x[1], self.range_y[0], self.range_y[1], self.range_y[2]))
                if key in self.metrics:
                    for place_id, score in self.metrics[key]:
                        top[test_id].setdefault(place_id, 0)
                        top[test_id][place_id] += score
                else:
                    log("The ({} ----> {}) of {} is not in metrics".format(test_x, key, test_id), DEBUG)

                    count_missing += 1

            log("The missing ratio is {}/{}={:4f}".format(count_missing, len(top), float(count_missing)/len(top)), INFO)

            if not self.is_testing:
                save_cache(top, filepath)

        return top

class KDTreeThread(BaseCalculatorThread):
    def process(self, test_ids, test_xs):
        stamp = make_a_stamp(str(test_ids) + str(test_xs))
        filepath = os.path.join(self.cache_workspace, "{}.pkl".format(stamp))

        top = load_cache(filepath)
        if not top or self.is_testing:
            top = {}

            distance, ind = self.metrics.query(test_xs, k=min(self.n_top, len(self.mapping)))
            for idx, loc in enumerate(ind):
                test_id = test_ids[idx]

                top.setdefault(test_id, {})
                for loc_idx, locc in enumerate(loc):
                    place_id = self.mapping[locc]

                    top[test_id].setdefault(place_id, 0)

                    d = distance[idx][loc_idx]
                    if d != 0:
                        top[test_id][place_id] += -1.0*np.log(distance[idx][loc_idx])*self.score[locc]

            if not self.is_testing:
                save_cache(top, filepath)

        return top

class ProcessThread(BaseCalculatorThread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        threading.Thread.__init__(self, group=group, target=target, name=name, verbose=verbose)

        self.args = args

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.strategy_engine = StrategyEngine(self.is_accuracy, self.is_exclude_outlier, self.is_testing, self.strategy)

    def run(self):
        while True:
            timestamp_start = time.time()

            filepath_train = self.queue.get()
            filename = os.path.basename(filepath_train)
            folder = os.path.dirname(filepath_train)

            filepath_train_pkl = os.path.join(os.path.dirname(self.cache_workspace), "train", "{}.{}.pkl".format(self.strategy, make_a_stamp(filepath_train)))
            create_folder(filepath_train_pkl)

            filepath_test = filepath_train.replace("train", "test")
            filepath_submission = os.path.join(self.submission_workspace, "{}.{}.pkl".format(type(self).__name__.lower(), filename))

            metrics, mapping = None, None
            if self.method == self.strategy_engine.STRATEGY_MOST_POPULAR:
                f = os.path.join(self.cache_workspace, "{}.{}.pkl".format(self.strategy_engine.get_most_popular_metrics.__name__.lower(), filename))
                metrics, (min_x, len_x), (min_y, len_y) = self.strategy_engine.get_most_popular_metrics(filepath_train, filepath_train_pkl, f, self.n_top, self.criteria[0], self.criteria[1], max(1, int(self.n_jobs*0.3)))
            elif self.method == self.strategy_engine.STRATEGY_KDTREE:
                f = os.path.join(self.cache_workspace, "{}.{}.pkl".format(self.strategy_engine.get_kdtree.__name__.lower(), filename))
                metrics, mapping, score = self.strategy_engine.get_kdtree(filepath_train, filepath_train_pkl, f, self.n_top, max(1, int(self.n_jobs*0.3)))
            else:
                log("Not implement this method, {}".format(self.method), ERROR)
                raise NotImplementError

            df = pd.read_csv(filepath_test, dtype={"row_id": np.int, "x":np.float, "y":np.float, "accuracy": np.int, "time": np.int})
            if self.is_testing:
                df = df.head(100)
            log("There are {} reocrds in {}".format(df.values.shape, filepath_test), INFO)

            fields = ["x", "y"]

            test_id = df["row_id"].values
            test_x = df[fields].values

            queue = Queue.Queue()
            for idx in range(0, test_id.shape[0]/self.batch_size+1):
                idx_start, idx_end = idx*self.batch_size, min(test_id.shape[0], (idx+1)*self.batch_size)
                size = test_id[idx_start:idx_end].shape[0]

                if size > 0:
                    queue.put((test_id[idx_start:idx_end], test_x[idx_start:idx_end]))
                else:
                    log("The batch size is {}({} - {})".format(size, idx_start, idx_end), ERROR)

            n_threads = max(1, self.n_jobs/2)
            for idx in range(0, n_threads):
                thread = None
                if self.method == self.strategy_engine.STRATEGY_MOST_POPULAR:
                    thread = MostPopularThread(kwargs={"queue": queue,
                                                       "results": self.results,
                                                       "cache_workspace": self.cache_workspace,
                                                       "metrics": metrics,
                                                       "transformer": self.strategy_engine.position_transformer,
                                                       "range_x": (min_x, len_x, self.criteria[0]),
                                                       "range_y": (min_y, len_y, self.criteria[1]),
                                                       "is_testing": self.is_testing,
                                                       "n_top": self.n_top})
                elif self.method == self.strategy_engine.STRATEGY_KDTREE:
                    thread = KDTreeThread(kwargs={"queue": queue,
                                                  "results": self.results,
                                                  "cache_workspace": self.cache_workspace,
                                                  "metrics": metrics,
                                                  "mapping": mapping,
                                                  "score": score,
                                                  "is_testing": self.is_testing,
                                                  "n_top": self.n_top})
                else:
                    log("Not implement this method, {}".format(self.method), ERROR)
                    raise NotImplementError

                thread.setDaemon(True)
                thread.start()

            queue.join()

            self.queue.task_done()

            timestamp_end = time.time()
            log("Cost {:8f} seconds to finish the prediction of {} by {}".format(timestamp_end-timestamp_start, filepath_train, self.method), INFO)

def process(method, workspaces, filepath_pkl, batch_size, criteria, strategy, is_accuracy, is_exclude_outlier, is_testing, n_top=3, n_jobs=8, count_testing_dataset=8607230):
    workspace, cache_workspace, output_workspace = workspaces
    for folder in [os.path.join(cache_workspace, "1.txt"), os.path.join(output_workspace, "1.txt")]:
        create_folder(folder)

    queue = Queue.Queue()
    for filename in scandir(os.path.join(workspace)):
        filepath_train = filename.path

        if filepath_train.find(".csv") != -1 and filepath_train.find("test.csv") == -1 and filepath_train.find("submission") == -1:
            # Avoid the empty file
            if os.stat(filepath_train).st_size > 238:
                queue.put(filepath_train)
                log("Push {} in queue".format(filepath_train), INFO)

    log("There are {} files in queue".format(queue.qsize()), INFO)

    threads, results = [], {}
    for idx in range(0, n_jobs):
        thread = ProcessThread(kwargs={"queue": queue,
                                       "results": results,
                                       "method": method,
                                       "criteria": criteria,
                                       "strategy": strategy,
                                       "batch_size": batch_size,
                                       "cache_workspace": cache_workspace,
                                       "submission_workspace": output_workspace,
                                       "is_accuracy": is_accuracy,
                                       "is_exclude_outlier": is_exclude_outlier,
                                       "is_testing": is_testing,
                                       "n_top": n_top,
                                       "n_jobs": n_jobs})
        thread.setDaemon(True)
        thread.start()

        threads.append(thread)
    queue.join()

    log("There are {} records in results".format(len(results)), INFO)

    timestamp_start = time.time()
    csv_format = {}
    for test_id, rankings in results.items():
        csv_format.setdefault(test_id, [])

        for place_id, most_popular in nlargest(n_top, sorted(rankings.items()), key=lambda (k, v): v):
            csv_format[test_id].append(str(int(place_id)))

        csv_format[test_id] = " ".join(csv_format[test_id])

    timestamp_end = time.time()
    log("Cost {:8f} secends to transform the results to submission".format(timestamp_end-timestamp_start), INFO)

    return csv_format

def save_submission(filepath, results, n_top=3):
    for test_id, info in results.items():
        results[test_id] = " ".join(info.split(" ")[:n_top])

    pd.DataFrame(results.items(), columns=["row_id", "place_id"]).to_csv(filepath, index=False, compression="gzip")

    log("The submission file is stored in {}".format(filepath), INFO)

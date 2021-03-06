# coding=UTF-8

import os
import sys
import glob
import copy
import time
import random
import socket

import numpy as np
import pandas as pd

from Queue import Queue
from threading import Thread, current_thread

from sklearn.linear_model import LinearRegression, Ridge, Lasso, RandomizedLasso
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler, Imputer
from sklearn.ensemble import RandomForestRegressor

from information_discrete import mi_3d, mi_4d

from load import save_cache, load_cache, load_interaction_information
from utils import log, DEBUG, INFO, WARN

class FeatureProfile(object):
    queue = Queue()
    ranking = {}

    def __init__(self):
        pass

    @staticmethod
    def normalization(ranks, names, order=1):
        if np.isnan(ranks).any():
            log("Found {} NaN values, so try to transform them to 'mean'".format(np.isnan(ranks).sum()), WARN)

            imp = Imputer(missing_values='NaN', strategy='mean', axis=1)
            imp.fit(ranks)
            ranks = imp.transform(ranks)[0]

        minmax = MinMaxScaler()
        r = minmax.fit_transform(order*np.array([ranks]).T).T[0]
        r = map(lambda x: round(x, 8), r)

        return dict(zip(names, r))

    @staticmethod
    def run(names, X, Y, filepath):
        filepath = filepath.replace(".pkl", "_{}.pkl".format(current_thread().name))

        def glob_ranking(folder):
            ranking = {}

            for f in glob.iglob("{}*.pkl".format(folder)):
                ranking.update(load_cache(f))

            return ranking

        while True:
            timestamp_start = time.time()
            done_ranking = glob_ranking(filepath)

            key, model, coef, order = FeatureProfile.queue.get()
            if key in done_ranking:
                log("{} was done before".format(key), INFO)
            else:
                if key == "Corr.":
                    f, pval = f_regression(X, Y, center=True)
                    FeatureProfile.ranking[key] = FeatureProfile.normalization(f, names)
                else:
                    model.fit(X, Y)
                    FeatureProfile.ranking[key] = FeatureProfile.normalization(np.abs(getattr(model, coef)), names, order)

                save_cache(copy.deepcopy(FeatureProfile.ranking), filepath)

            timestamp_end = time.time()
            log("Cost {:.4f} secends to finish {}".format(time.time() - timestamp_start, key), INFO)

            FeatureProfile.queue.task_done()

    def profile(self, X, Y, names, filepath, score_function, n_features_rfe=5):
        # create parent folder
        folder_parent = os.path.dirname(filepath)
        if not os.path.isdir(folder_parent):
            os.makedirs(folder_parent)

        lr = LinearRegression(normalize=True)
        FeatureProfile.queue.put(("Linear Reg.", lr, "coef_", 1))

        ridge = Ridge(alpha=7)
        FeatureProfile.queue.put(("Ridge", ridge, "coef_", 1))

        lasso = Lasso(alpha=.05)
        FeatureProfile.queue.put(("Lasso", lasso, "coef_", 1))

        rlasso = RandomizedLasso(alpha=0.04)
        FeatureProfile.queue.put(("Stability", rlasso, "scores_", 1))

        #stop the search when 5 features are left (they will get equal scores)
        rfe = RFE(lr, n_features_to_select=n_features_rfe)
        FeatureProfile.queue.put(("RFE", rfe, "ranking_", -1))

        rf = RandomForestRegressor(n_jobs=-1)
        FeatureProfile.queue.put(("RF", rf, "feature_importances_", 1))

        FeatureProfile.queue.put(("Corr.", None, None, 1))

        for idx in range(0, 4):
            thread = Thread(target=FeatureProfile.run, kwargs={"names": names, "X":X, "Y":Y, "filepath": "{}.pkl".format(filepath)})
            thread.setDaemon(True)
            thread.start()

        FeatureProfile.queue.join()

        r = {}
        for name in names:
            r[name] = round(np.mean([FeatureProfile.ranking[method][name] for method in FeatureProfile.ranking.keys()]), 8)

        methods = sorted(FeatureProfile.ranking.keys())
        FeatureProfile.ranking["Mean"] = r
        methods.append("Mean")

        FeatureProfile.ranking["Feature"] = dict(zip(names, names))

        pd.DataFrame(FeatureProfile.ranking).to_csv("{}.csv".format(filepath), index=False)
        save_cache(FeatureProfile.ranking, "{}.pkl".format(filepath))

        return FeatureProfile.ranking

class InteractionInformation(object):
    def __init__(self, dataset, train_y, folder_couple, combinations_size=2):
        self.dataset = dataset
        self.train_y = train_y

        self.folder_couple = folder_couple
        self.results_couple = {}

        self.queue = Queue()

        self.combinations_size = combinations_size

        self.read_cache()

    def read_cache(self):
        if not os.path.isdir(self.folder_couple):
            os.makedirs(self.folder_couple)

        for filepath_couple in glob.iglob("{}/*pkl*".format(self.folder_couple)):
            try:
                obj = load_cache(filepath_couple)

                if obj:
                    self.results_couple.update(obj)
            except ValueError as e:
                pass

        log("Already finish {} results_couple".format(len(self.results_couple)))

    def compose_column_names(self, names):
        return "{};target".format(";".join(names))

    def decompose_column_names(self, name):
        return name.split(";")[:-1]

    def add_item(self, column_names, combinations_size):
        key = self.compose_column_names(column_names)

        if key not in self.results_couple:
            self.queue.put((key, combinations_size))

            if self.queue.qsize() % 10000 == 0:
                log("Put I({}) into the queue, and the size of queue is {}".format(key, self.queue.qsize()), INFO)

class InteractionInformationThread(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        Thread.__init__(self, group=group, target=target, name=name, verbose=verbose)

        self.args = args
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_couple(self, key, v):
        self.results_couple[key] = v
        self.dump()

    def dump(self, force_dump=False):
        if force_dump or len(self.results_couple) > self.batch_size_dump:
            if len(self.results_couple) > 0:
                filepath_couple = "{}/{}.{}.pkl".format(self.folder_couple, socket.gethostname(), int(10000*time.time()))
                log("write {} records in {}".format(len(self.results_couple), filepath_couple), INFO)
                save_cache(self.results_couple, filepath_couple)
                self.results_couple = {}
            else:
                log("The self.results_couple is empty", INFO)

    def run(self):
        while True:
            timestamp_start = time.time()
            column_couple, combinations_size = self.ii.queue.get()
            column_names = self.ii.decompose_column_names(column_couple)

            mi = -1
            if combinations_size == 1:
                mi = mi(self.ii.dataset[column_names[0]].values, self.ii.train_y.values)
            elif combinations_size == 2:
                mi = mi_3d((column_names[0], self.ii.dataset[column_names[0]].values),
                            (column_names[1], self.ii.dataset[column_names[1]].values),
                            ("target", self.ii.train_y.values))
            elif combinations_size == 3:
                mi = mi_4d((column_names[0], self.ii.dataset[column_names[0]].values),
                            (column_names[1], self.ii.dataset[column_names[1]].values),
                            (column_names[2], self.ii.dataset[column_names[2]].values),
                            ("target", self.ii.train_y.values))

            self.add_couple(column_couple, mi)

            timestamp_end = time.time()

            if mi > 2e-04:
                log("Cost {:.2f} secends to calculate I({}) is {:.6f}, the remaining size is {}/{}".format(timestamp_end-timestamp_start, column_couple, mi, self.ii.queue.qsize(), len(self.results_couple)), INFO)
            elif self.ii.queue.qsize() % 10000 == 0:
                log("The remaining size of self.ii.queue is {}".format(self.ii.queue.qsize()), INFO)

            self.ii.queue.task_done()

        # Dump the results
        log("Force dumpping the results", INFO)
        self.dump(True)

# coding=UTF-8

import os
import sys
import glob
import time
import pprint
import random

import re
import numpy as np
import pandas as pd

import pandas.core.algorithms as algos

from Queue import Queue
from threading import Thread, current_thread

from itertools import combinations
from collections import Counter

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RandomizedLasso
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler, Imputer
from sklearn.ensemble import RandomForestRegressor
from minepy import MINE
from information_discrete import mi_3d, mi_4d

from load import save_cache, load_cache, load_interaction_information
from utils import log, DEBUG, INFO, WARN

class FeatureProfile(object):
    def __init__(self):
        pass

    def normalization(slef, ranks, names, order=1):
        if np.isnan(ranks).any():
            log("Found {} NaN values, so try to transform them to 'mean'".format(np.isnan(ranks).sum()), WARN)

            imp = Imputer(missing_values='NaN', strategy='mean', axis=1)
            imp.fit(ranks)
            ranks = imp.transform(ranks)[0]

        minmax = MinMaxScaler()
        r = minmax.fit_transform(order*np.array([ranks]).T).T[0]
        r = map(lambda x: round(x, 8), r)

        return dict(zip(names, r))

    def profile(self, X, Y, names, filepath, n_features_rfe=5):
        ranks = {}

        timestamp_start = time.time()
        lr = LinearRegression(normalize=True)
        lr.fit(X, Y)
        ranks["Linear reg"] = self.normalization(np.abs(lr.coef_), names)
        log("Cost {:.4f} secends to finish Linear Regression".format(time.time() - timestamp_start), INFO)

        timestamp_start = time.time()
        ridge = Ridge(alpha=7)
        ridge.fit(X, Y)
        ranks["Ridge"] = self.normalization(np.abs(ridge.coef_), names)
        log("Cost {:.4f} secends to finish Ridge".format(time.time() - timestamp_start), INFO)

        timestamp_start = time.time()
        lasso = Lasso(alpha=.05)
        lasso.fit(X, Y)
        ranks["Lasso"] = self.normalization(np.abs(lasso.coef_), names)
        log("Cost {:.4f} secends to finish Lasso".format(time.time() - timestamp_start), INFO)

        timestamp_start = time.time()
        rlasso = RandomizedLasso(alpha=0.04)
        rlasso.fit(X, Y)
        ranks["Stability"] = self.normalization(np.abs(rlasso.scores_), names)
        log("Cost {:.4f} secends to finish Stability".format(time.time() - timestamp_start), INFO)

        #stop the search when 5 features are left (they will get equal scores)
        timestamp_start = time.time()
        rfe = RFE(lr, n_features_to_select=n_features_rfe)
        rfe.fit(X,Y)
        ranks["RFE"] = self.normalization(map(float, rfe.ranking_), names, order=-1)
        log("Cost {:.4f} secends to finish RFE".format(time.time() - timestamp_start), INFO)

        timestamp_start = time.time()
        rf = RandomForestRegressor()
        rf.fit(X,Y)
        ranks["RF"] = self.normalization(rf.feature_importances_, names)
        log("Cost {:.4f} secends to finish Random Forest".format(time.time() - timestamp_start), INFO)

        timestamp_start = time.time()
        f, pval = f_regression(X, Y, center=True)
        ranks["Corr."] = self.normalization(f, names)
        log("Cost {:.4f} secends to finish Corr.".format(time.time() - timestamp_start), INFO)

        '''
        timestamp_start = time.time()
        mine = MINE()
        mic_scores = []
        for i in range(X.shape[1]):
            mine.compute_score(X[:,i], Y)
            m = mine.mic()
            mic_scores.append(m)

        ranks["MIC"] = self.normalization(mic_scores, names)
        '''

        r = {}
        for name in names:
            r[name] = round(np.mean([ranks[method][name] for method in ranks.keys()]), 8)
        log("Cost {:.4f} secends to finish MIC".format(time.time() - timestamp_start), INFO)

        methods = sorted(ranks.keys())
        ranks["Mean"] = r
        methods.append("Mean")

        ranks["Feature"] = dict(zip(names, names))

        '''
        print "\t%s" % "\t".join(methods)
        for name in names:
            print "%s\t%s" % (name, "\t".join(map(str, [ranks[method][name] for method in methods])))
        '''

        pd.DataFrame(ranks).to_csv(filepath, index=False)

        return ranks

def transform(distribution):
    keys = distribution.keys()
    values = distribution.values()
    total = sum(values)
    values = map(lambda x: x/float(total), values)

    return keys, values

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
            log("Try to load cache file from {}".format(filepath_couple))

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
            filepath_couple = "{}/{}.pkl".format(self.folder_couple, int(10000*time.time()))

            save_cache(self.results_couple, filepath_couple)
            self.results_couple = {}

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
        self.dump(True)

def load_dataset(filepath_cache, dataset, binsize=2, threshold=0.1):
    LABELS = "abcdefghijklmnopqrstuvwxABCDEFGHIJKLMNOPQRSTUVWX0123456789!@#$%^&*()_+~"
    FIXED_LABELS = "yYz"

    def less_replace(df, c, unique_values, labels):
        for i, unique_value in enumerate(unique_values):
            df[c][df[c] == unique_value] == labels[i]

    drop_columns = []
    if os.path.exists(filepath_cache):
        dataset = load_cache(filepath_cache)
    else:
        count_raw = len(dataset[dataset.columns[0]].values)
        for idx, column in enumerate(dataset.columns):
            data_type = dataset.dtypes[idx]
            unique_values = dataset[column].unique()

            try:
                if column != "target":
                    if data_type == "object":
                        if len(unique_values) < len(LABELS):
                            less_replace(dataset, column, unique_values, LABELS)
                            log("Change {} by unique type(size={})".format(column, len(unique_values)), INFO)
                        else:
                            log("The size of {} is too large({})".format(column, len(unique_values)), WARN)
                    else:
                        is_break = False
                        deleted_idxs = np.array([])
                        counter = Counter(dataset[column].values).most_common(len(FIXED_LABELS))
                        for idx_label, (name, value) in enumerate(counter):
                            ratio = float(value) / count_raw

                            if ratio == 1:
                                drop_columns.append(column)
                                log("The size of most common value of {} is 1 so skipping it".format(column), INFO)

                                is_break = True
                                break
                            elif ratio > threshold:
                                log("The ratio of common value({}, {}) of {} is {}, greater".format(data_type, name, column, ratio), INFO)

                                idxs_most_common = np.where(dataset[column] == name)[0]
                                deleted_idxs = np.concatenate((deleted_idxs, idxs_most_common), axis=0)

                                dataset[column][idxs_most_common] = FIXED_LABELS[idx_label]
                            else:
                                log("The ratio of common value({}, {}) of {} is {}, less".format(data_type, name, column, ratio), INFO)

                                break

                        if is_break:
                            continue
                        else:
                            ori_idxs = np.array([tmp_i for tmp_i in range(0, count_raw)])
                            idxs_non_most_common = np.delete(ori_idxs, deleted_idxs)

                            non_common_unique_values = dataset[column][idxs_non_most_common].unique()

                            if len(non_common_unique_values) < len(LABELS):
                                for ii, unique_value in enumerate(non_common_unique_values):
                                    dataset[column][dataset[column] == unique_value] = LABELS[ii]
                            else:
                                is_success = False
                                for tmp_binsize in [t for t in range(len(LABELS)-1, 0, -4)]:
                                    try:
                                        tmp = pd.qcut(dataset[column][idxs_non_most_common].values, tmp_binsize, labels=[c for c in LABELS[:tmp_binsize]])
                                        dataset[column][idxs_non_most_common] = tmp
                                        is_success = True

                                        break
                                    except ValueError as e:
                                        if e.message.find("Bin edges must be unique") > -1:
                                            log("Descrease binsize from {} to {} for {} again due to {}".format(column, tmp_binsize, tmp_binsize-4, str(e)), DEBUG)
                                        else:
                                            raise

                                if is_success:
                                    log("The final binsize of {} is {}".format(column, tmp_binsize), INFO)
                                else:
                                    log("Fail in transforming {}".format(column), WARN)
                                    drop_columns.append(column)

                                    continue

                            log("Change {} by bucket type".format(column), INFO)

                    dataset[column] = ["Z" if str(value) == "nan" else value for value in dataset[column]]
                else:
                    log("The type of {} is already categorical".format(column), INFO)
            except ValueError as e:
                log("The size of unique values of {} is {}, greater than {}".format(column, len(unique_values), binsize), INFO)
                raise

        dataset = dataset.drop(drop_columns, axis=1)

        dataset.to_csv("{}.csv".format(filepath_cache))
        save_cache(dataset, filepath_cache)

    return dataset

def calculate_interaction_information(filepath_cache, dataset, train_y, folder_couple, combinations_size,
                                      n_split_idx=0, n_split_num=1, binsize=2, nthread=4, is_testing=None):
    dataset = load_dataset(filepath_cache, dataset, binsize)

    ii = InteractionInformation(dataset, train_y, folder_couple, combinations_size)

    count_break = 0

    for size in range(combinations_size, 1, -1):
        rounds = list(combinations([column for column in dataset.columns], size))
        for pair_column in rounds[n_split_idx::n_split_num]:
            if is_testing and random.random()*10 > 1: # Random Sampling when is_testing = True
                continue

            ii.add_item(pair_column, size)

            if is_testing and count_break > is_testing:
                log("Early break due to the is_testing is True", INFO)
                break
            else:
                count_break += 1

    # Memory Concern
    ii.results_couple = {}

    for idx in range(0, nthread):
        worker = InteractionInformationThread(kwargs={"ii": ii, "results_couple": {}, "folder_couple": folder_couple, "batch_size_dump": 2**13})
        worker.setDaemon(True)
        worker.start()

    log("Wait for the completion of the calculation of Interaction Information", INFO)
    ii.queue.join()

    return ii.results_couple

def test_new_interaction_information(filepath_cache, dataset, train_y, binsize=4):
    import math
    from information_discrete import mi

    dataset = load_dataset(filepath_cache, dataset, binsize)
    # Cost 162.86 secends to calculate I(var3;imp_op_var39_comer_ult1;saldo_medio_var5_hace3;target) is 0.030339899738, the remaining size is 776080
    #tmp_df = dataset["var3"]
    #tmp_df["imp_op_var39_comer_ult1"] = dataset["imp_op_var39_comer_ult1"]
    #tmp_df["saldo_medio_var5_hace3"] = dataset["saldo_medio_var5_hace3"]
    #tmp_df["target"] = train_y.values

    # ind_var34;saldo_medio_var5_ult3;target 0.0149167532518
    a = mi(dataset["ind_var34"].values, dataset["saldo_medio_var5_ult3"].values)
    b = mi_3d(("ind_var34", dataset["ind_var34"].values), ("saldo_medio_var5_ult3", dataset["saldo_medio_var5_ult3"].values), ("target", train_y.values))

    distribution = {}
    x = np.unique(dataset["ind_var34"].values)
    y = np.unique(dataset["saldo_medio_var5_ult3"].values)
    z = np.unique(train_y.values)

    from entropy_estimators import cmidd
    c = cmidd(dataset["ind_var34"].values, dataset["saldo_medio_var5_ult3"].values, train_y.values)

    print c, a, b, c-a

    print mi_4d(("var3", dataset["var3"].values), ("ind_var34", dataset["ind_var34"].values), ("saldo_medio_var5_ult3", dataset["saldo_medio_var5_ult3"].values), ("target", train_y.values))

def merge_binsize(filepath_output, pattern, topX=500):
    dfs = []

    for filepath in glob.glob(pattern):
        if filepath.find("cache") == -1:
            filename = os.path.basename(filepath)
            binsize = re.search("(binsize=(\d+))", filename).groups()[0]

            index, series = [], {"{}_value".format(binsize): [], "{}_rank".format(binsize): []}
            for key, value in load_interaction_information(filepath, topX):
                index.append(";".join(key))
                series["{}_value".format(binsize)].append(value)
                series["{}_rank".format(binsize)].append(len(index))

            dfs.append(pd.DataFrame(series, index=index))

    # Merge
    results = pd.concat(dfs, axis=1)
    results.to_csv(filepath_output)

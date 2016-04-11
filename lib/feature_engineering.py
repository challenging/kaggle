# coding=UTF-8

import os
import sys
import glob
import time
import pprint
import random

import re
import dit
import numpy as np
import pandas as pd

import pandas.core.algorithms as algos

from Queue import Queue
from threading import Thread, Lock

from itertools import combinations
from scipy.stats import mode

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RandomizedLasso
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler, Imputer
from sklearn.ensemble import RandomForestRegressor
from minepy import MINE

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
    def __init__(self, dataset, train_y, filepath_couple, filepath_series, filepath_criteria, combinations_size=2, threshold=0.01, save_size=100):
        self.dataset = dataset
        self.train_y = train_y

        self.filepath_couple = filepath_couple
        self.filepath_series = filepath_series
        self.filepath_criteria = filepath_criteria

        self.queue = Queue()
        self.lock_couple = Lock()

        self.combinations_size = combinations_size
        self.threshold = threshold
        self.ori_save_size = save_size
        self.save_size = save_size

        self.cache_series = {}
        self.cache_criteria = {}

        self.results_couple = {}

        self.read_cache()

    def read_cache(self):
        if os.path.exists(self.filepath_couple):
            self.results_couple = load_cache(self.filepath_couple)

        if os.path.exists(self.filepath_series):
            self.cache_series = load_cache(self.filepath_series)

        if os.path.exists(self.filepath_criteria):
            self.cache_criteria = load_cache(self.filepath_criteria)

    def write_cache(self, results=False):
        if results:
            save_cache(self.results_couple, self.filepath_couple)
        else:
            save_cache(self.cache_series, self.filepath_series)
            save_cache(self.cache_criteria, self.filepath_criteria)

    def get_values_from_cache(self, column_x):
        a, b = None, None

        if column_x not in self.cache_series:
            self.cache_series[column_x] = pd.Series(self.dataset[column_x])
        a = self.cache_series[column_x]

        if column_x not in self.cache_criteria:
            self.cache_criteria[column_x] = self.dataset[column_x].unique()
        b = self.cache_criteria[column_x]

        return (a, b)

    def compose_column_names(self, names):
        return "{};target".format(";".join(names))

    def decompose_column_names(self, name):
        return name.split(";")[:-1]

    def add_item(self, column_names):
        key = self.compose_column_names(column_names)

        if key not in self.results_couple:
            self.queue.put(key)
            log("Put I({}) into the queue".format(key), INFO)
        else:
            log("I({}) is done".format(key), INFO)

    def add_couple(self, key ,v):
        with self.lock_couple:
            self.results_couple[key] = v

            if self.save_size < 0:
                self.write_cache(True)

                self.save_size = self.ori_save_size
            else:
                self.save_size -= 1

class InteractionInformationThread(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        Thread.__init__(self, group=group, target=target, name=name, verbose=verbose)

        self.args = args
        for key, value in kwargs.items():
            setattr(self, key, value)

    def run(self):
        LABELS = ["A", "B", "C", "D", "E"]
        series_target = pd.Series(self.ii.train_y.astype(str))

        while True:
            timestamp_start = time.time()
            column_couple = self.ii.queue.get()
            column_names = self.ii.decompose_column_names(column_couple)

            criteria_index, criteria_value = [], []

            series = {"target": series_target}
            for name in column_names:
                series_value, column_criteria = self.ii.get_values_from_cache(name)
                series[name] = series_value

                criteria_index.append(len(criteria_index))
                criteria_value.append((name, column_criteria))

            tmp_df = pd.DataFrame(series)

            distribution = {}
            for idxs in combinations(criteria_index, self.ii.combinations_size):
                for criteria_target in ["0", "1"]:
                    if self.ii.combinations_size == 2:
                        layer1, layer2 = idxs[0], idxs[1]
                        layer1_name, layer1_values = criteria_value[layer1]
                        layer2_name, layer2_values = criteria_value[layer2]

                        for layer1_value in layer1_values:
                            for layer2_value in layer2_values:
                                key = "{}{}{}".format(layer1_value, layer2_value, criteria_target)
                                distribution[key] = len(np.where((tmp_df[layer1_name] == layer1_value) & (tmp_df[layer2_name] == layer2_value) & (tmp_df["target"] == criteria_target))[0])
                    elif self.ii.combinations_size == 3:
                        layer1, layer2, layer3 = idxs[0], idxs[1], idxs[2]
                        layer1_name, layer1_values = criteria_value[layer1]
                        layer2_name, layer2_values = criteria_value[layer2]
                        layer3_name, layer3_values = criteria_value[layer3]

                        for layer1_value in layer1_values:
                            for layer2_value in layer2_values:
                                for layer3_value in layer3_values:
                                    key = "{}{}{}{}".format(layer1_value, layer2_value, layer3_value, criteria_target)
                                    distribution[key] = len(np.where((tmp_df[layer1_name] == layer1_value) & (tmp_df[layer2_name] == layer2_value) & (tmp_df[layer3_name] == layer3_value)  & (tmp_df["target"] == criteria_target))[0])
                    elif self.ii.combinations_size == 4:
                        layer1, layer2, layer3, layer4 = idxs[0], idxs[1], idxs[2], idxs[3]
                        layer1_name, layer1_values = criteria_value[layer1]
                        layer2_name, layer2_values = criteria_value[layer2]
                        layer3_name, layer3_values = criteria_value[layer3]
                        layer4_name, layer4_values = criteria_value[layer4]

                        for layer1_value in layer1_values:
                            for layer2_value in layer2_values:
                                for layer3_value in layer3_values:
                                    for layer4_value in layer4_values:
                                        key = "{}{}{}{}{}".format(layer1_value, layer2_value, layer3_value, layer4_value, criteria_target)
                                        distribution[key] = len(np.where((tmp_df[layer1_name] == layer1_value) & (tmp_df[layer2_name] == layer2_value) & (tmp_df[layer3_name] == layer3_value) & (tmp_df[layer4_name] == layer4_value) & (tmp_df["target"] == criteria_target))[0])
                    else:
                        log("Not support the combination size is greater than 4", WARN)
                        self.ii.queue_task_done()

                        continue

            keys, values = transform(distribution)

            mi = dit.Distribution(keys, values)

            interaction_information = -1

            rv_names = LABELS[:self.ii.combinations_size]
            mi.set_rv_names(rv_names + ["Z"])
            interaction_information = dit.shannon.mutual_information(mi, rv_names, ["Z"])

            self.ii.add_couple(column_couple, interaction_information)

            timestamp_end = time.time()
            if interaction_information >= self.ii.threshold:
                log("Cost {:.2f} secends to calculate I({}) is {}, the remaining size is {}".format(timestamp_end-timestamp_start, column_couple, interaction_information, self.ii.queue.qsize()), INFO)

            self.ii.queue.task_done()

def load_dataset(filepath_cache, dataset, binsize=2):
    LABELS = "abcdefghijklmnopqrstuvwxyABCDEFGHIJKLMNOPQRSTUVWXY0123456789!@#$%^&*()_+~"

    def less_replace(df, c, unique_values, labels):
        for i, unique_value in enumerate(unique_values):
            df[c][df[c] == unique_value] == labels[i]

    idxs = []
    if os.path.exists(filepath_cache):
        idxs, dataset = load_cache(filepath_cache)
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
                        most_commons = mode(dataset[column].values, axis=None)
                        most_common_value, most_common_count = most_commons[0][0], most_commons[1][0]

                        if float(most_common_count) / count_raw >= 0.3:
                            dataset[column][dataset[column] == most_common_value] = "z"
                            idxs_most_common = np.where(dataset[column][dataset[column] == most_common_value])[0]
                            non_common_unique_values = np.unique(dataset[column][~idxs_most_common])
                            if len(non_common_unique_values) < len(LABELS):
                                for ii, unique_value in enumerate(non_common_unique_values):
                                    dataset[column][dataset[column] == unique_value] = LABELS[ii]
                            else:
                                dataset[column][~idxs_most_common] = pd.qcut(dataset[column].values[~idx_most_common], binsize, labels=[c for c in LABELS[:binsize]])
                        else:
                            dataset[column] = pd.qcut(dataset[column].values, binsize, labels=[c for c in LABELS[:binsize]])

                        log("Change {} by bucket type".format(column), INFO)

                    dataset[column] = ["Z" if str(value) == "nan" else value for value in dataset[column]]

                    idxs.append(idx)
                else:
                    log("The type of {} is already categorical".format(column), INFO)
            except ValueError as e:
                log("The size of unique values of {} is {}, greater than {}".format(column, len(unique_values), len(LABELS)), INFO)
                raise

        save_cache((idxs, dataset), filepath_cache)

    return idxs, dataset

def calculate_interaction_information(filepath_cache, dataset, train_y, filepath_couple, filepah_series, filepath_criteria, combinations_size,
                                      binsize=2, threshold=0.01, nthread=4, is_testing=None):
    idxs, dataset = load_dataset(filepath_cache, dataset, binsize)

    ii = InteractionInformation(dataset, train_y, filepath_couple, filepah_series, filepath_criteria, combinations_size, threshold)

    # Build Cache File
    timestamp_start = time.time()
    for column_x_idx in range(0, len(idxs)):
        column_x = dataset.columns[idxs[column_x_idx]]
        ii.get_values_from_cache(column_x)
    ii.write_cache()
    timestamp_end = time.time()
    log("Cost {:.4f} secends to build cache files".format(timestamp_end-timestamp_start), INFO)

    count_break = 0
    for pair in combinations(idxs, combinations_size):
        if is_testing and random.random()*10 > 1:
            continue

        column_names = [dataset.columns[idx] for idx in pair]

        ii.add_item(column_names)

        if is_testing and count_break > is_testing:
            log("Early break due to the is_testing is True", INFO)
            break
        else:
            count_break += 1

    for idx in range(0, nthread):
        worker = InteractionInformationThread(kwargs={"ii": ii})
        worker.setDaemon(True)
        worker.start()

    log("Wait for the completion of the calculation of Interaction Information", INFO)
    ii.queue.join()

    log("Write the results", INFO)
    ii.write_cache(results=True)

    return ii.results_couple

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

if __name__ == "__main__":
    merge_binsize("../input/merge_binsize.csv", "../input/transform2*testing=-1*type=2*binsize=*pkl", 1000)

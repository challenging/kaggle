import os
import sys
import time
import pprint

import dit
import numpy as np
import pandas as pd

from Queue import Queue
from threading import Thread, Lock

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RandomizedLasso
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler, Imputer
from sklearn.ensemble import RandomForestRegressor
from minepy import MINE

from load import save_cache, load_cache
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
    def __init__(self, dataset, train_y, filepath_couple, filepath_single, threshold=0.01, save_size=100):
        self.dataset = dataset
        self.train_y = train_y

        self.filepath_couple = filepath_couple
        self.filepath_single = filepath_single

        self.queue = Queue()
        self.lock_single = Lock()
        self.lock_couple = Lock()
        self.lock_s = Lock()
        self.lock_c = Lock()

        self.threshold = threshold
        self.ori_save_size = save_size
        self.save_size = save_size

        self.cache_series = {}
        self.cache_criteria = {}

        self.results_single = {}
        self.results_couple = {}

    def read_cache(self):
        if os.path.exists(self.filepath_single):
            self.results_single = load_cache(self.filepath_single)

        if os.path.exists(self.filepath_couple):
            self.results_couple = load_cache(self.filepath_couple)

    def get_values_from_cache(self, column_x):
        a, b = None, None

        if column_x not in self.cache_series:
            with self.lock_s:
                self.cache_series[column_x] = pd.Series(self.dataset[column_x])
        a = self.cache_series[column_x]

        if column_x not in self.cache_criteria:
            with self.lock_c:
                self.cache_criteria[column_x] = self.dataset[column_x].unique()
        b = self.cache_criteria[column_x]

        return (a, b)

    def add_item(self, column_x, column_y=None):
        if column_y:
            if column_x not in self.results_couple or column_y not in self.results_couple[column_x]:
                self.queue.put((column_x, column_y))
                log("Put I({};{};target) into the queue".format(column_x, column_y), INFO)
            else:
                log("I({};{};target) is done".format(column_x, column_y), INFO)
        else:
            if column_x not in self.results_single:
                self.queue.put((column_x, column_y))
            else:
                log("I({};target) is done".format(column_x), INFO)

    def add_single(self, x, v):
        with self.lock_single:
            self.results_single[x] = v

            if self.save_size < 0:
                save_cache(self.results_single, self.filepath_single)
                self.save_size = self.ori_save_size
            else:
                self.save_size -= 1

    def add_couple(self, x, y ,v):
        with self.lock_couple:
            self.results_couple.setdefault(x, {})
            self.results_couple[x][y] = v

            if self.save_size < 0:
                save_cache(self.results_couple, self.filepath_couple)
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
        series_target = pd.Series(self.ii.train_y.astype(str))

        while True:
            timestamp_start = time.time()
            (column_x, column_y) = self.ii.queue.get()

            interaction_information = -1
            if column_y:
                series_x, column_x_criteria = self.ii.get_values_from_cache(column_x)
                series_y, column_y_criteria = self.ii.get_values_from_cache(column_y)

                series = {column_x: series_x, column_y: series_y, "target": series_target}
                tmp_df = pd.DataFrame(series)

                distribution = {}
                for criteria_x in column_x_criteria:
                    for criteria_y in column_y_criteria:
                        for criteria_z in ["0", "1"]:
                            key = "{}{}{}".format(criteria_x, criteria_y, criteria_z)
                            distribution[key] = len(np.where((tmp_df[column_x] == criteria_x) & (tmp_df[column_y] == criteria_y) & (tmp_df["target"] == criteria_z))[0])
                keys, values = transform(distribution)

                mi = dit.Distribution(keys, values)
                mi.set_rv_names(["X", "Y", "Z"])
                interaction_information = dit.shannon.mutual_information(mi, ["X", "Y"], ["Z"])

                self.ii.add_couple(column_x, column_y, interaction_information)
            else:
                series_x, column_x_criteria = self.ii.get_values_from_cache(column_x)
                series = {column_x: series_x, "target": series_target}
                tmp_df = pd.DataFrame(series)

                distribution = {}
                for criteria_x in column_x_criteria:
                    for criteria_z in ["0", "1"]:
                        key = "{}{}".format(criteria_x, criteria_z)
                        distribution[key] = len(np.where((tmp_df[column_x] == criteria_x) & (tmp_df["target"] == criteria_z))[0])

                keys, values = transform(distribution)

                mi = dit.Distribution(keys, values)
                mi.set_rv_names(["X", "Z"])
                interaction_information = dit.shannon.mutual_information(mi, ["X"], ["Z"])

                self.ii.add_single(column_x, interaction_information)

            timestamp_end = time.time()
            if interaction_information >= self.ii.threshold:
                if column_y:
                    log("Cost {:.2f} secends to calculate I({};{};target) is {}".format(timestamp_end-timestamp_start, column_x, column_y, interaction_information), INFO)
                else:
                    log("Cost {:.2f} secends to calculate I({};target) is {}".format(timestamp_end-timestamp_start, column_x, interaction_information), INFO)
            else:
                if column_y:
                    log("Cost {:.2f} secends to calculate I({};{};target) is {}".format(timestamp_end-timestamp_start, column_x, column_y, interaction_information), DEBUG)
                else:
                    log("Cost {:.2f} secends to calculate I({};target) is {}".format(timestamp_end-timestamp_start, column_x, interaction_information), DEBUG)

            self.ii.queue.task_done()

def load_dataset(filepath_cache, dataset, binsize=2):
    LABELS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXY0123456789!@#$%^&*()_+~"

    idxs = []
    if os.path.exists(filepath_cache):
        idxs, dataset = load_cache(filepath_cache)
    else:
        for idx, column in enumerate(dataset.columns):
            data_type = dataset.dtypes[idx]
            unique_values = dataset[column].unique()

            try:
                if data_type != "object" and column != "target":
                    if len(unique_values) < len(LABELS):
                        for i, unique_value in enumerate(unique_values):
                            dataset[column][dataset[column] == unique_value] = LABELS[i]
                        log("Change {} by unique type".format(column), INFO)
                    else:
                        dataset[column] = pd.qcut(dataset[column].values, binsize, labels=[c for c in LABELS[:binsize]])
                        log("Change {} by bucket type".format(column), INFO)

                    dataset[column] = ["Z" if str(value) == "nan" else value for value in dataset[column]]

                    idxs.append(idx)
                else:
                    log("The type of {} is already categorical".format(column), INFO)
            except ValueError as e:
                log("The size of unique values of {} is {}, greater than {}".format(column, len(unique_values), len(LABELS)), INFO)

        save_cache((idxs, dataset), filepath_cache)

    return idxs, dataset

def calculate_interaction_information(filepath_cache, dataset, train_y, filepath_couple, filepath_single, binsize=2, threshold=0.01, nthread=4, is_testing=None):
    idxs, dataset = load_dataset(filepath_cache, dataset, binsize)

    ii = InteractionInformation(dataset, train_y, filepath_couple, filepath_single, threshold)

    for column_x_idx in range(0, len(idxs)):
        column_x = dataset.columns[idxs[column_x_idx]]
        column_x_criteria = dataset[column_x].unique()

        timestamp_start_x = time.time()
        for column_y_idx in range(column_x_idx+1, len(idxs)):
            column_y = dataset.columns[idxs[column_y_idx]]

            ii.add_item(column_x, column_y)

        ii.add_item((column_x, None))

        if is_testing and column_x_idx > is_testing:
            log("Early break due to the is_testing is True", INFO)
            break

    for idx in range(0, nthread):
        worker = InteractionInformationThread(kwargs={"ii": ii})
        worker.setDaemon(True)
        worker.start()

    log("Wait for the completion of the calculation of Interaction Information", INFO)
    ii.queue.join()

    return ii.results_single, ii.results_couple

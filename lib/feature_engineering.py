import os
import sys
import time

import dit
import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RandomizedLasso
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler, Imputer
from sklearn.ensemble import RandomForestRegressor
from minepy import MINE

sys.path.append("{}/../lib".format(os.path.dirname(os.path.abspath(__file__))))
from utils import log, INFO, WARN

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

def interaction_information(dataset, train_y, binsize=2, threshold=0.01):
    LABELS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXY0123456789!@#$%^&*()_+~"

    idxs = []
    for idx, column in enumerate(dataset.columns):
        log("Try to process {}".format(column), INFO)

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
        except ValueError as e:
            log("The size of unique values of {} is {}, greater than {}".format(column, len(unique_values), len(LABELS)), INFO)

    results_single, results_couple = {}, {}
    size = len(dataset.values)
    for i in range(0, len(idxs)):
        column_x = dataset.columns[idxs[i]]

        timestamp_start_x = time.time()
        for ii in range(i+1, len(idxs)):
            timestamp_start = time.time()

            distribution = {}
            column_y = dataset.columns[idxs[ii]]

            series = {column_x: pd.Series(dataset[column_x]),
                      column_y: pd.Series(dataset[column_y]),
                      "target": pd.Series(train_y.astype(str))}

            tmp_df = pd.DataFrame(series)

            for criteria_x in list(LABELS[:binsize]) + ["Z"]:
                for criteria_y in list(LABELS[:binsize]) + ["Z"]:
                    for criteria_z in ["0", "1"]:
                        key = "{}{}{}".format(criteria_x, criteria_y, criteria_z)
                        distribution[key] = len(np.where((tmp_df[column_x] == criteria_x) & (tmp_df[column_y] == criteria_y) & (tmp_df["target"] == criteria_z))[0])

            #pprint.pprint(distribution)

            keys = distribution.keys()
            values = distribution.values()
            total = sum(values)
            values = map(lambda x: x/float(total), values)

            mi = dit.Distribution(keys, values)
            mi.set_rv_names(["X", "Y", "Z"])
            interaction_information = dit.shannon.mutual_information(mi, ["X", "Y"], ["Z"])
            timestamp_end = time.time()

            if interaction_information >= threshold:
                log("Cost {:.2f} secends to calculate I({};{};target) is {}".format(timestamp_end-timestamp_start, column_x, column_y, interaction_information), INFO)

            results_couple["I({};{};target)".format(column_x, column_y)] = interaction_information

        mi = dit.shannon.mutual_information(mi, ["X"], ["Z"])

        timestamp_end_x = time.time()
        log("{:.2f} secends, I({};target) is {}".format(timestamp_end_x-timestamp_start_x, column_x, mi), INFO)

        results_single["I({};target)".format(column_x)] = mi

    return results_single, results_couple

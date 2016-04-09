#!/usr/bin/env python

import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pylab as plt

from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.metrics import log_loss, make_scorer

from utils import log, INFO, WARN
from load import load_data, data_transform_2, load_cache, save_cache, load_interaction_information

BASEPATH = os.path.dirname(os.path.abspath(__file__))

def score_function(y, y_hat):
    flog_loss_ = log_loss(y, y_hat)

class ParameterTuning(object):
    def __init__(self, target, data_id, method, n_estimator, cost, objective, cv, n_jobs):
        self.target = target
        self.data_id = data_id
        self.method = method

        self.n_estimator = n_estimator
        self.cost = cost
        self.objective = objective
        self.cv = cv
        self.n_jobs = n_jobs

        self.random_state = 1201

        self.best_cost = -np.inf

        self.train = None
        self.done = {}

    def set_filepath(self, filepath):
        self.filepath = filepath

    def set_train(self, train):
        self.train = train

        predictors = [x for x in self.train.columns if x not in [self.target, self.data_id]]
        self.predictors = predictors

    def compare(self, cost):
        if cost > self.best_cost:
            self.best_cost = cost

            return True
        elif cost == self.best_cost:
            log("The cost is the same with the previous cost - {}".format(cost))

            return True
        else:
            return False

    def get_best_params(self, grid_model, x, y):
        grid_model.fit(x, y)

        return grid_model.best_score_, grid_model.best_params_, grid_model.grid_scores_

    def improve(self, phase, cost, params, micro_tuning=False):
        old_cost = self.best_cost

        if self.compare(cost):
            log("Improve the cost from {} to {}".format(old_cost, self.best_cost))
            for key, value in params.items():
                setattr(self, key, value)
                log("Set {} to be {}".format(key, getattr(self, key)))

            save_cache(self, self.filepath)
        else:
            if not micro_tuning:
                log("Fail so terminate due to the cost of phase2-model is {}(> {}), and the params is {}".format(cost, old_cost, params), WARN)

                sys.exit(1)

    def get_value(self, name):
        return getattr(self, name) if getattr(self, name) else getattr(self, "default_{}".format(name))

    def get_model_instance(self):
        raise NotImeplementError

    def phase(self, phase, params, is_micro_tuning=False):
        gsearch1 = None
        if phase in self.done:
            log("The {} is done so we skip it".format(phase))
            for key in params.keys():
                log("The {} is {} based on {}".format(key, getattr(self, key)), phase)

            best_cost, best_params, scores, gsearch1 = self.done[phase]
        else:
            gsearch1 = GridSearchCV(estimator=self.get_model_instance(),
                                    param_grid=params,
                                    scoring=make_scorer(log_loss),
                                    n_jobs=self.n_jobs,
                                    iid=False,
                                    cv=self.cv,
                                    verbose=1)

            best_cost, best_params, scores = self.get_best_params(gsearch1, self.train[self.predictors], self.train[self.target])
            log("The cost of {}-model is {:.8f} based on {}".format(phase, best_cost, best_params.keys()))
            self.improve(phase, best_cost, best_params)

            self.done[phase] = best_cost, best_params, scores, gsearch1

        gsearch2 = None
        micro_cost, micro_params, micro_scores = -np.inf, -np.inf, None
        if is_micro_tuning:
            key = "micro-{}".format(phase)
            if key in self.done:
                log("The {} is done so we skip it".format(key))
                for name in params.keys():
                    log("The {} is {} based on {}".format(name, getattr(self, name)), key)

                micro_cost, micro_params, micro_scores, gsearch2 = self.done[key]
            else:
                advanced_params = {}
                for name, value in best_params.items():
                    if isinstance(value, int):
                        advanced_params[name] = [i for i in range(max(0, value-1), value+1)]
                    elif isinstance(value, float):
                        advanced_params[name] = [value*i for i in [0.5, 1, 5, 0]]

                gsearch2 = GridSearchCV(estimator=self.get_model_instance(),
                                        param_grid=advanced_params,
                                        scoring=self.cost,
                                        n_jobs=self.n_jobs,
                                        iid=False,
                                        cv=self.cv,
                                        verbose=1)

                micro_cost, micro_params, micro_scores = self.get_best_params(gsearch2, self.train[self.predictors], self.train[self.target])
                log("Finish the micro-tuning of {}, and then get best params is {}".format(phase, micro_params))
                self.improve(key, micro_cost, micro_params, True)

                self.done[key] = micro_cost, micro_params, micro_scores, gsearch2

        model = None
        a, b, c = None, None, None
        if micro_cost > best_cost:
            model = gsearch2
            a, b, c = micro_cost, micro_params, micro_scores
        else:
            model = gsearch1
            a, b, c = best_cost, best_params, scores

        if self.method == "classifier":
            predicted_proba = model.predict_proba(self.train[self.predictors])[:,1]
            logloss = log_loss(self.train[self.target], predicted_proba)
            log("The {} of training dataset is {:.8f}".format(self.cost, logloss))
        elif self.method == "regressor":
            predicted_proba = model.predict(self.train[self.predictors])
            logloss = log_loss(self.train[self.target], predicted_proba)
            log("The {} of training dataset is {:.8f}".format(self.cost, logloss))
        else:
            log("???? {}".format(self.method))

        return a, b, c

    def process(self):
        raise NotImplementError

class RandomForestTuning(ParameterTuning):
    def __init__(self, target, data_id, method, n_estimator=200, cost="log_loss", objective="entropy", cv=5, n_jobs=-1):
        ParameterTuning.__init__(self, target, data_id, method, n_estimator, cost, objective, cv, n_jobs)

        self.default_criterion, self.criterion = "entropy", None
        self.default_max_features, self.max_features = 0.5, None
        self.default_max_depth, self.max_depth = 8, None
        self.default_min_samples_split, self.min_samples_split = 4, None
        self.default_min_samples_leaf, self.min_samples_leaf = 2, None
        self.default_class_weight, self.class_weight = {0: 1, 1: 1}, None

    def get_model_instance(self):
        n_estimator = self.get_value("n_estimator")

        criterion = self.get_value("criterion")
        max_features = self.get_value("max_features")
        max_depth = self.get_value("max_depth")

        if self.method == "classifier":
            return RandomForestClassifier(n_estimators=n_estimator,
                                          #criterion=criterion,
                                          max_features=max_features,
                                          max_depth=max_depth)
        elif self.method == "regressor":
            return RandomForestRegressor(n_estimators=n_estimator,
                                         #criterion=criterion,
                                         max_features=max_features,
                                         max_depth=max_depth)

    def process(self):
        phase1_cost, phase1_params, phase1_scores = self.phase("phase1", {})

        param2 = {'max_depth': range(6, 16, 2), 'max_features': [ratio for ratio in [0.25, 0.5, 0.75]], "min_samples_leaf": range(2, 10, 2), "min_samples_split": range(4, 10, 2)}

        if self.method == "classifier":
            param2["class_weight"] = [{0: 1, 1: 1}, {0: 1.5, 1: 1}, {0: 2, 1: 1}, "balanced"]

        phase2_cost, phase2_params, phase2_scores = self.phase("phase2", param2)

class ExtraTreeTuning(RandomForestTuning):
    pass

class XGBoostingTuning(ParameterTuning):
    def __init__(self, target, data_id, method, n_estimator=200, cost="log_loss", objective="binary:logistic", cv=5, n_jobs=-1):
        ParameterTuning.__init__(self, target, data_id, method, n_estimator, cost, objective, cv, n_jobs)

        self.default_learning_rate, self.learning_rate = 0.1, None
        self.default_max_depth, self.max_depth = 5, None
        self.default_min_child_weight, self.min_child_weight = 1, None

        self.default_gamma, self.gamma = 0, None
        self.default_subsample, self.subsample = 0.8, None
        self.default_colsample_bytree, self.colsample_bytree = 0.8, None

        self.default_reg_alpha, self.reg_alpha = 0, None

    def get_model_instance(self):
        learning_rate = self.get_value("learning_rate")
        n_estimator = self.get_value("n_estimator")
        max_depth = self.get_value("max_depth")
        min_child_weight = self.get_value("min_child_weight")
        gamma = self.get_value("gamma")
        subsample = self.get_value("subsample")
        colsample_bytree = self.get_value("colsample_bytree")
        reg_alpha = self.get_value("reg_alpha")

        if self.method == "classifier":
            log("Current parameters - learning_rate: {}, n_estimator: {}, max_depth: {}, min_child_weight: {}, gamma: {}, subsample: {}, colsample_bytree: {}, reg_alpha: {}".format(learning_rate, n_estimator, max_depth, min_child_weight, gamma, subsample, colsample_bytree, reg_alpha))

            return xbg.XGBClassifier(learning_rate=learning_rate,
                                     n_estimators=n_estimator,
                                     max_depth=max_depth,
                                     min_child_weight=min_child_weight,
                                     gamma=gamma,
                                     subsample=subsample,
                                     colsample_bytree=colsample_bytree,
                                     reg_alpha=reg_alpha,
                                     objective=self.objective,
                                     nthread=self.n_jobs,
                                     scale_pos_weight=1,
                                     seed=self.random_state)

        elif self.method == "regressor":
            return xgb.XGBRegressor(learning_rate=learning_rate,
                                    n_estimators=n_estimator,
                                    max_depth=max_depth,
                                    min_child_weight=min_child_weight,
                                    gamma=gamma,
                                    subsample=subsample,
                                    colsample_bytree=colsample_bytree,
                                    reg_alpha=reg_alpha,
                                    objective=self.objective,
                                    nthread=self.n_jobs,
                                    scale_pos_weight=1,
                                    seed=self.random_state)

    def process(self):
        phase1_cost, phase1_params, phase1_scores = self.phase("phase1", {})

        param2 = {'max_depth':range(3,16,2), 'min_child_weight':range(1, 10, 2)}
        phase2_cost, phase2_params, phase2_scores = self.phase("phase2", param2, True)

        param3 = {'gamma':[i/10.0 for i in range(0,5)]}
        phase3_cost, phase3_params, phase3_scores = self.phase("phase3", param3)

        param4 = {'subsample':[i/10.0 for i in range(6, 10)], 'colsample_bytree':[i/10.0 for i in range(6, 10)]}
        phase4_cost, phase4_params, phase4_scores = self.phase("phase4", param4)

        param5 = {'reg_alpha':[1e-5, 1e-2, 0.1, 1.0, 100.0]}
        phase5_cost, phase5_params, phase5_scores = self.phase("phase5", param5, True)
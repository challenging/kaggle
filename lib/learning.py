#!/usr/bin/env python

import os
import sys
import numpy as np
import xgboost as xgb

import time
import Queue
import copy
import pickle
import threading

# For Deep Learning
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint

# For Shadow Learning
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import log_loss, make_scorer

sys.path.append("{}/../lib".format(os.path.dirname(os.path.abspath(__file__))))
from utils import log, DEBUG, INFO, WARN, ERROR
from deep_learning import logistic_regression, KaggleCheckpoint

class LearningFactory(object):
    n_jobs = -1
    ensemble_params = {"ne": 200,"md": 6,"mf": 80,"rs": 1201}

    @staticmethod
    def set_n_estimators(n_estimators):
        LearningFactory.ensemble_params["ne"] = n_estimators

    @staticmethod
    def cost_function(y, y_hat):
        flog_loss_ = log_loss(y, y_hat) #, eps=1e-15, normalize=True, sample_weight=None)
        return flog_loss_

    @staticmethod
    def get_model(method):
        model = None
        LL = make_scorer(LearningFactory.cost_function, greater_is_better=False)

        if method.find("shallow") > -1:
            if method.find("linear_regressor") > -1:
                gs = GridSearchCV(estimator=LinearRegression(), param_grid={}, n_jobs=LearningFactory.n_jobs, cv=2, verbose=0, scoring=LL)

                model = Learning(method, gs)
            elif method.find("logistic_regressor") > -1:
                params = {"C": [1e-04, 1e-02, 1e-01, 1, 1e+01, 1e+02, 1e+04], "solver": ["newton-cg", "lbfgs", "liblinear"]}

                model = Learning(method, GridSearchCV(LogisticRegression(), params, verbose=1, scoring=LL))
            elif method.find("regressor") > -1:
                if method.find("extratree") > -1:
                    gs = GridSearchCV(ExtraTreesRegressor(n_estimators=LearningFactory.ensemble_params["ne"],
                                                          max_depth=LearningFactory.ensemble_params["md"],
                                                          max_features=LearningFactory.ensemble_params["mf"],
                                                          random_state=LearningFactory.ensemble_params["rs"],
                                                          n_jobs=-1),
                                      param_grid={}, n_jobs=LearningFactory.n_jobs, cv=2, verbose=0, scoring=LL)
                    model = Learning(method, gs)
                elif method.find("randomforest") > -1:
                    gs = GridSearchCV(RandomForestRegressor(n_estimators=LearningFactory.ensemble_params["ne"],
                                                            max_depth=LearningFactory.ensemble_params["md"],
                                                            max_features=LearningFactory.ensemble_params["mf"],
                                                            random_state=LearningFactory.ensemble_params["rs"],
                                                            min_samples_split=4,
                                                            min_samples_leaf=2,
                                                            verbose=0,
                                                            n_jobs=-1),
                                      param_grid={}, n_jobs=LearningFactory.n_jobs, cv=2, verbose=0, scoring=LL)
                    model = Learning(method, gs)
                elif method.find("gradientboosting") > -1:
                    gs = GridSearchCV(GradientBoostingRegressor(n_estimators=LearningFactory.ensemble_params["ne"],
                                                                max_depth=LearningFactory.ensemble_params["md"],
                                                                max_features=LearningFactory.ensemble_params["mf"],
                                                                random_state=LearningFactory.ensemble_params["rs"],
                                                                learning_rate=1e-01),
                                      param_grid={}, n_jobs=LearningFactory.n_jobs, cv=2, verbose=0, scoring=LL)

                    model = Learning(method, gs)
                elif method.find("xgboosting") > -1:
                    model = Learning(method, xgb.XGBRegressor(n_estimators=LearningFactory.ensemble_params["ne"],
                                                              max_depth=LearningFactory.ensemble_params["md"],
                                                              seed=LearningFactory.ensemble_params["rs"],
                                                              missing=np.nan, learning_rate=1e-02, subsample=0.9, colsample_bytree=0.85, objective="reg:linear"))
            elif method.find("classifier") > -1:
                if method.find("extratree") > -1:
                    gs = GridSearchCV(ExtraTreesClassifier(n_estimators=LearningFactory.ensemble_params["ne"],
                                                           max_depth=LearningFactory.ensemble_params["md"],
                                                           max_features=LearningFactory.ensemble_params["mf"],
                                                           random_state=LearningFactory.ensemble_params["rs"],
                                                           n_jobs=-1),
                                      param_grid={}, n_jobs=LearningFactory.n_jobs, cv=2, verbose=0, scoring=LL)
                    model = Learning(method, gs)
                elif method.find("randomforest") > -1:
                    gs = GridSearchCV(RandomForestClassifier(n_estimators=LearningFactory.ensemble_params["ne"],
                                                             max_depth=LearningFactory.ensemble_params["md"],
                                                             max_features=LearningFactory.ensemble_params["mf"],
                                                             random_state=LearningFactory.ensemble_params["rs"],
                                                             criterion="entropy",
                                                             min_samples_split=4, min_samples_leaf=2, verbose=0, n_jobs=-1),
                                      param_grid={}, n_jobs=LearningFactory.n_jobs, cv=2, verbose=0, scoring=LL)
                    model = Learning(method, gs)
                elif method.find("gradientboosting") > -1:
                    gs = GridSearchCV(GradientBoostingClassifier(n_estimators=LearningFactory.ensemble_params["ne"],
                                                                 max_depth=LearningFactory.ensemble_params["md"],
                                                                 max_features=LearningFactory.ensemble_params["mf"],
                                                                 random_state=LearningFactory.ensemble_params["rs"],
                                                                 learning_rate=1e-01),
                                      param_grid={}, n_jobs=LearningFactory.n_jobs, cv=2, verbose=0, scoring=LL)
                    model = Learning(method, gs)
                elif method.find("xgboosting") > -1:
                    model = Learning(method, xgb.XGBClassifier(n_estimators=LearningFactory.ensemble_params["ne"],
                                                              max_depth=LearningFactory.ensemble_params["md"],
                                                              seed=LearningFactory.ensemble_params["rs"],
                                                              missing=np.nan, learning_rate=1e-02, subsample=0.9, colsample_bytree=0.85, objective="binary:logistic"))
            elif method.find("svm") > -1:
                    model = Learning(method, SVC(probability=True, random_state=LearningFactory.ensemble_params["rs"]))
            else:
                log("Error model naming - {}".format(method), WARN)
        elif method.find("deep") > -1:
            model = Learning(method, None)

        return model

class Learning(object):
    def __init__(self, name, model):
        self.name = name.lower()
        self.model = model

    def init_deep_params(self, model_folder, layer, mini_batch, dimension, train_x, train_y, number_of_feature,
                         nepoch, callbacks=[]):
        self.nepoch = nepoch
        self.callbacks = callbacks

        self.model = logistic_regression(model_folder, layer, mini_batch, dimension, train_x, train_y, number_of_feature)

    def is_shallow_learning(self):
        return self.name.find("shallow") != -1

    def is_deep_learning(self):
        return self.name.find("deep") != -1

    def is_xgb(self):
        return self.name.find("xgb") != -1

    def is_regressor(self):
        return self.name.find("regressor") != -1

    def is_classifier(self):
        return self.name.find("classifier") != -1

    def is_svm(self):
        return self.name.find("svm") != -1

    def is_grid_search(self):
        return self.name.find("gridsearch") != -1

    def train(self, train_x, train_y):
        if self.is_shallow_learning():
            if self.is_xgb():
                self.model.fit(train_x, train_y, eval_metric="logloss")
            else:
                self.model.fit(train_x, train_y)
        elif self.is_deep_learning():
            self.model.fit(train_x, train_y, nb_epoch=self.nepoch, batch_size=self.mini_batch, callbacks=self.callbacks)

    def predict(self, data):
        if self.is_regressor():
            return self.model.predict(data)
        elif self.is_classifier():
            # Only care the probability of class '1'
            return self.model.predict_proba(data)[:,1]
        elif self.is_svm():
            return self.model.predict_proba(data)[:,1]

    def grid_scores(self):
        if self.is_grid_search():
            return self.model.grid_scores_
        else:
            return None

    def cost(self, data, y_true):
        return log_loss(y_true, self.predict(data))

    def coef(self):
        if self.is_grid_search():
            return self.model.best_estimator_.coef_ if hasattr(self.model.best_estimator_, "coef_") else np.nan
        else:
            return self.model.coef_ if hasattr(self.model, "coef_") else np.nan

class LearningLogLoss(object):
    def __init__(self, models, nfold):
        self.logloss = {}
        for model in models:
            self.logloss.setdefault(model, [0.0 for idx in range(0, nfold)])

    def insert_logloss(self, model_name, nfold, cost):
        self.logloss[model_name][nfold] += cost

class LearningQueue(object):
    def __init__(self, train_x, train_y, test_x, filepath=None):
        self.lock = threading.Lock()
        self.learning_queue = Queue.Queue()

        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.filepath = filepath

    def setup_layer_info(self, layer_two_training_dataset, layer_two_testing_dataset, learning_logloss):
        self.layer_two_training_dataset = layer_two_training_dataset
        self.layer_two_testing_dataset = layer_two_testing_dataset
        self.learning_logloss = learning_logloss

    def put(self, nfold, model_idx, dataset_idxs, model):
        self.learning_queue.put((nfold, model_idx, dataset_idxs, model))

    def starts(self, number_of_thread=1):
        for idx in range(0, number_of_thread):
            worker = LearningThread(kwargs={"obj": self})
            worker.setDaemon(True)
            worker.start()

        self.learning_queue.join()

    def is_done_layer_two_training_dataset(self, layer_two_training_idx, model_idx):
        if np.sum(self.layer_two_training_dataset[layer_two_training_idx, model_idx] == 0.0) > 0.5*len(self.layer_two_training_dataset[layer_two_training_idx, model_idx]):
            return False
        else:
            return True

    def insert_layer_two_training_dataset(self, layer_two_training_idx, model_idx, results):
        self.lock.acquire()

        try:
            self.layer_two_training_dataset[layer_two_training_idx, model_idx] = results
        finally:
            self.lock.release()

    def insert_layer_two_testing_dataset(self, model_idx, nfold, results):
        self.lock.acquire()

        try:
            self.layer_two_testing_dataset[:, model_idx, nfold] = results
        finally:
            self.lock.release()

    def dump(self):
        self.lock.acquire()

        try:
            if self.filepath:
                folder = os.path.dirname(self.filepath)
                if not os.path.isdir(folder):
                    os.makedirs(folder)

                objs = (self.layer_two_training_dataset, self.layer_two_testing_dataset, self.learning_logloss)
                with open(self.filepath, "wb") as OUTPUT:
                    pickle.dump(objs, OUTPUT)

                log("Save queue in {}".format(self.filepath), DEBUG)
            else:
                log("Not set the filepath to save", WARN)
        finally:
            self.lock.release()

class LearningThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        threading.Thread.__init__(self, group=group, target=target, name=name, verbose=verbose)

        self.args = args

        for key, value in kwargs.items():
            setattr(self, key, value)

    def run(self):
        while True:
            (nfold, model_idx, (train_x_idx, test_x_idx), model_name) = self.obj.learning_queue.get()
            timestamp_start = time.time()

            model = LearningFactory.get_model(model_name)
            if model == None:
                log("Can't init this model({})".format(model_name), WARN)
            else:
                model.train(self.obj.train_x[train_x_idx], self.obj.train_y[train_x_idx])

                results = model.predict(self.obj.train_x[test_x_idx])
                self.obj.insert_layer_two_training_dataset(test_x_idx, model_idx, results)

                layer_two_testing_dataset= None
                if model.is_regressor():
                    layer_two_testing_dataset = model.predict(self.obj.test_x)
                elif model.is_classifier():
                    layer_two_testing_dataset = model.predict(self.obj.test_x)

                self.obj.insert_layer_two_testing_dataset(model_idx, nfold, layer_two_testing_dataset)

                cost = model.cost(self.obj.train_x[test_x_idx], self.obj.train_y[test_x_idx])
                if np.isnan(cost):
                    log("The cost of '{}' model for {}th fold is NaN".format(model_name, nfold), WARN)
                else:
                    self.obj.learning_logloss.insert_logloss(model_name, nfold, cost)

                timestamp_end = time.time()
                log("The grid score is {}".format(model.grid_scores()), DEBUG)
                log("Cost {:02f} secends to train '{}' model for fold-{:02d}, and the logloss is {:.8f}, the best score is {:.8f}".format(\
                        timestamp_end-timestamp_start, model.name, nfold, cost, model.coef()), INFO)

            self.obj.learning_queue.task_done()
            self.obj.dump()

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
from sklearn.metrics import auc, log_loss, make_scorer

# For Cluster
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, AgglomerativeClustering, SpectralClustering
from sklearn.neighbors import NearestCentroid, kneighbors_graph
from sklearn.preprocessing import MinMaxScaler, Imputer, StandardScaler

from load import load_advanced_data, save_cache
from utils import log, DEBUG, INFO, WARN, ERROR
from deep_learning import logistic_regression, logistic_regression_2, KaggleCheckpoint
from customized_estimators import CustomizedClassEstimator, CustomizedProbaEstimator

BASEPATH = os.path.dirname(os.path.abspath(__file__))

class LearningFactory(object):
    @staticmethod
    def get_model(pair, cost_function=log_loss):
        model = None
        method, setting = pair

        LL = make_scorer(cost_function)

        log("Try to create model based on {}".format(method), INFO)
        if method.find("shallow") > -1:
            if method.find("linear_regressor") > -1:
                model = Learning(method, LinearRegression(), cost_function)
            elif method.find("logistic_regressor") > -1:
                model = Learning(method, LogisticRegression())
            elif method.find("regressor") > -1:
                if method.find("extratree") > -1:
                    model = Learning(method, ExtraTreesRegressor(**setting), cost_function)
                elif method.find("randomforest") > -1:
                    model = Learning(method, RandomForestRegressor(**setting), cost_function)
                elif method.find("gradientboosting") > -1:
                    model = Learning(method, GradientBoostingRegressor(**setting), cost_function)
                elif method.find("xgboosting") > -1:
                    model = Learning(method, xgb.XGBRegressor(**setting), cost_function)
                else:
                    log("1. Can't create model based on {}".format(method), ERROR)
            elif method.find("classifier") > -1:
                if method.find("extratree") > -1:
                    model = Learning(method, ExtraTreesClassifier(**setting), cost_function)
                elif method.find("randomforest") > -1:
                    model = Learning(method, RandomForestClassifier(**setting), cost_function)
                elif method.find("gradientboosting") > -1:
                    model = Learning(method, GradientBoostingClassifier(**setting), cost_function)
                elif method.find("xgboosting") > -1:
                    model = Learning(method, xgb.XGBClassifier(**setting), cost_function)
                else:
                   log("2. Can't create model based on {}".format(method), ERROR)
            else:
                log("3. Can't create model based on {}".format(method), ERROR)
        elif method.find("cluster") > -1:
            if method.find("kmeans") > -1:
                model = Learning(method, KMeans(**setting))
            else:
                log("4. Can't create model based on {}".format(method), ERROR)
        elif method.find("deep") > -1:
            setting["folder"] = "{}/nn_layer={}_neurno={}_{}th".format(setting["folder"], setting["number_of_layer"], setting["dimension"], setting["nfold"])
            if not os.path.isdir(setting["folder"]):
                try:
                    os.makedirs(setting["folder"])
                except OSError as e:
                    pass

            log("The folder of deep learning is in {}".format(setting["folder"]), INFO)

            model = Learning(method, None, cost_function)
            model.init_deep_params(**setting)
        elif method.find("customized") > -1:
            if method.find("class") > -1:
                model = Learning(method, CustomizedClassEstimator(**setting), cost_function)
            elif method.find("proba") > -1:
                model = Learning(method, CustomizedProbaEstimator(**setting), cost_function)
            else:
                log("5. Can't create model based on {}".format(method), ERROR)
        else:
            log("6. Can't create model based on {}".format(method), ERROR)

        return model

class Learning(object):
    def __init__(self, name, model, cost_function=log_loss):
        self.name = name.lower()
        self.model = model
        self.cost_function = cost_function

    def init_deep_params(self, nfold, folder, input_dims,
                         number_of_layer, batch_size, dimension,
                         nepoch, validation_split, class_weight, callbacks=[]):

        self.batch_size = batch_size
        self.nepoch = nepoch

        self.callbacks = callbacks
        self.callbacks[-1].folder = folder

        self.class_weight = class_weight
        self.validation_split = validation_split

        self.model = logistic_regression(folder, number_of_layer, dimension, input_dims)

    def is_shallow_learning(self):
        return self.name.find("shallow") > -1

    def is_deep_learning(self):
        return self.name.find("deep") > -1

    def is_xgb(self):
        return self.name.find("xgb") > -1

    def is_regressor(self):
        return self.name.find("regressor") > -1

    def is_classifier(self):
        return self.name.find("classifier") > -1

    def is_svm(self):
        return self.name.find("svm") > -1

    def is_cluster(self):
        return self.name.find("cluster") > -1

    def is_grid_search(self):
        return self.name.find("gridsearch") > -1

    def get_labels(self):
        if self.is_cluster():
            return self.model.labels_
        else:
            return None

    def get_cluster_results(self, train_x, test_x):
        labels = self.get_labels()

        training_results, testing_results = [], []

        training_labels = self.predict(train_x)
        for label in training_labels:
            training_results.append(self.ratio[label])

        testing_labels = self.predict(test_x)
        for label in testing_labels:
            testing_results.append(self.ratio[label])

        return training_results, testing_results

    def train(self, train_x, train_y):
        if self.is_cluster():
            train_x = train_x.astype(float) - train_x.min(0) / train_x.ptp(axis=0)
            if np.isnan(train_x).any():
                log("Found {} NaN values, so try to transform them to 'mean'".format(np.isnan(train_x).sum()), WARN)

                imp = Imputer(missing_values='NaN', strategy='mean', axis=1)
                imp.fit(train_x)

                train_x = imp.transform(train_x)

            self.model.fit(train_x)
            labels = self.get_labels()

            ratio = {}
            for idx, target in enumerate(train_y):
                label = labels[idx]

                ratio.setdefault(label, [0, 0])
                ratio[label][int(target)] += 1

            for label, nums in ratio.items():
                target_0, target_1 = nums[0], nums[1]
                ratio[label] = float(target_1) / (target_0 + target_1)
            self.ratio = ratio
        elif self.is_shallow_learning():
            log("The parameters of {} are {}".format(self.name, self.model.get_params()), DEBUG)
            self.model.fit(train_x, train_y)
        elif self.is_deep_learning():
            self.model.fit(train_x, train_y, nb_epoch=self.nepoch, batch_size=self.batch_size, validation_split=self.validation_split, class_weight=self.class_weight, callbacks=self.callbacks)

    def predict(self, data):
        if self.is_shallow_learning():
            if self.is_regressor():
                if self.name.find("logistic_regressor") > -1:
                    return self.model.predict_proba(data)[:, 1]
                else:
                    return self.model.predict(data)
            elif self.is_classifier():
                # Only care the probability of class '1'
                return self.model.predict_proba(data)[:,1]
            elif self.is_svm():
                return self.model.predict_proba(data)[:,1]
        elif self.is_cluster():
            return self.model.predict(data)
        elif self.is_deep_learning():
            return [prob[0] if prob else 0.0 for prob in self.model.predict_proba(data)]

    def grid_scores(self):
        if self.is_grid_search():
            return self.model.grid_scores_
        else:
            return None

    def cost(self, data, y_true):
        return self.cost_function(y_true, self.predict(data))

    def coef(self):
        if self.is_shallow_learning():
            if self.is_grid_search():
                return self.model.best_estimator_.coef_ if hasattr(self.model.best_estimator_, "coef_") else np.nan
            else:
                return self.model.coef_ if hasattr(self.model, "coef_") else np.nan
        else:
            return self.model.get_weights()

class LearningCost(object):
    def __init__(self, models, nfold):
        self.cost = {}
        for model in models:
            self.cost.setdefault(model, np.zeros(nfold).astype(float))

    def insert_cost(self, model_name, nfold, cost):
        if model_name not in self.cost:
            self.cost.setdefault(model_name, np.zeros(len(self.cost.values()[0])))
            log("Not Found {} in self.cost, so creating it".format(model_name), WARN)

        self.cost[model_name][nfold] += cost

class LearningQueue(object):
    def __init__(self, train_x, train_y, test_x, filepath=None):
        self.lock = threading.Lock()
        self.learning_queue = Queue.Queue()

        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.filepath = filepath

    def setup_layer_info(self, layer_two_training_dataset, layer_two_testing_dataset, learning_cost):
        self.layer_two_training_dataset = layer_two_training_dataset
        self.layer_two_testing_dataset = layer_two_testing_dataset
        self.learning_cost = learning_cost

    def put(self, folder, nfold, model_idx, dataset_idxs, model):
        self.learning_queue.put((folder, nfold, model_idx, dataset_idxs, model))

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

    def insert_layer_two_training_dataset(self, layer_two_training_idx, model_name, model_idx, results, model_folder=None):
        self.lock.acquire()

        try:
            self.layer_two_training_dataset[layer_two_training_idx, model_idx] = results

            if model_folder:
                filepath = "{}/middle_layer/{}.dataset.pkl".format(model_folder, model_name)
                save_cache(self.layer_two_training_dataset[:,model_idx], filepath)
        finally:
            self.lock.release()

    def insert_layer_two_testing_dataset(self, model_name, model_idx, nfold, results, model_folder=None):
        self.lock.acquire()

        try:
            self.layer_two_testing_dataset[:, model_idx, nfold] = results

            if model_folder:
                filepath = "{}/middle_layer/{}.dataset.pkl".format(model_folder, model_name)
                save_cache(self.layer_two_testing_dataset[:, model_idx], filepath)
        finally:
            self.lock.release()

    def dump(self):
        self.lock.acquire()

        try:
            if self.filepath:
                folder = os.path.dirname(self.filepath)
                if not os.path.isdir(folder):
                    os.makedirs(folder)

                objs = (self.layer_two_training_dataset, self.layer_two_testing_dataset, self.learning_cost)
                save_cache(objs, self.filepath)

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
        log("{} starts...".format(threading.current_thread().name), INFO)

        while True:
            (model_folder, nfold, model_idx, (train_x_idx, test_x_idx), pair) = self.obj.learning_queue.get()
            timestamp_start = time.time()

            cost = -1
            model_name = pair[0]

            if model_name.find("deep") > -1:
                pair[1]["folder"] = model_folder
                pair[1]["nfold"] = nfold

            model = LearningFactory.get_model(pair)
            if model == None or model.model == None:
                log("Can't init this model({})".format(model_name), WARN)
            elif model.is_cluster():
                model.train(self.obj.train_x[train_x_idx], self.obj.train_y[train_x_idx])
                training_results, testing_results = model.get_cluster_results(self.obj.train_x[test_x_idx], self.obj.test_x)

                self.obj.insert_layer_two_training_dataset(test_x_idx, model.name, model_idx, training_results, model_folder)
                self.obj.insert_layer_two_testing_dataset(model.name, model_idx, nfold, testing_results, model_folder)
                self.obj.learning_cost.insert_cost(model_name, nfold, -1)
            else:
                model.train(self.obj.train_x[train_x_idx], self.obj.train_y[train_x_idx])

                results = model.predict(self.obj.train_x[test_x_idx])
                self.obj.insert_layer_two_training_dataset(test_x_idx, model.name, model_idx, results, model_folder)

                layer_two_testing_dataset = model.predict(self.obj.test_x)
                self.obj.insert_layer_two_testing_dataset(model.name, model_idx, nfold, layer_two_testing_dataset, model_folder)

                cost = model.cost(self.obj.train_x[test_x_idx], self.obj.train_y[test_x_idx])
                if np.isnan(cost):
                    log("The cost of '{}' model for {}th fold is NaN".format(model_name, nfold), WARN)
                else:
                    self.obj.learning_cost.insert_cost(model_name, nfold, cost)

                log("The grid score is {}".format(model.grid_scores()), DEBUG)

            timestamp_end = time.time()
            log("Cost {:02f} secends to train '{}' model for fold-{:02d}, and cost is {:.8f}".format(\
                    timestamp_end-timestamp_start, model.name, nfold, cost), INFO)

            self.obj.learning_queue.task_done()
            self.obj.dump()

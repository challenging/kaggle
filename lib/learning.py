#!/usr/bin/env python

import os
import sys
import md5
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

# For Shallow Learning
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score, log_loss

# For Cluster
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, AgglomerativeClustering, SpectralClustering
from sklearn.neighbors import NearestCentroid, kneighbors_graph
from sklearn.preprocessing import MinMaxScaler, Imputer, StandardScaler

from load import save_cache, load_cache
from utils import log, DEBUG, INFO, WARN, ERROR
from deep_learning import logistic_regression, logistic_regression_2, KaggleCheckpoint
from customized_estimators import CustomizedClassEstimator, CustomizedProbaEstimator

BASEPATH = os.path.dirname(os.path.abspath(__file__))

class LearningFactory(object):
    @staticmethod
    def get_model(objective, pair, cost_function):
        model = None
        method, setting = pair

        if isinstance(cost_function, str):
            if cost_function == "log_loss":
                cost_function = log_loss
            elif cost_function == "auc":
                cost_function = roc_auc_score
            else:
                log("Wrong cost_function - {}".format(cost_function), ERROR)
                sys.exit(100)

        log("Try to create model based on {}".format(method), INFO)
        if method.find("shallow") > -1:
            if method.find("logistic_regressor") > -1:
                if "cost" in setting:
                    del setting["cost"]

                model = Learning(objective, method, LogisticRegression(**setting), cost_function)
            elif method.find("linear_regressor") > -1:
                model = Learning(objective, method, LinearRegression(), cost_function)
            elif method.find("regressor") > -1:
                if method.find("extratree") > -1:
                    model = Learning(objective, method, ExtraTreesRegressor(**setting), cost_function)
                elif method.find("randomforest") > -1:
                    model = Learning(objective, method, RandomForestRegressor(**setting), cost_function)
                elif method.find("gradientboosting") > -1:
                    model = Learning(objective, method, GradientBoostingRegressor(**setting), cost_function)
                elif method.find("xgboosting") > -1:
                    if "n_jobs" in setting:
                        log("Delete n_jobs={} from the setting for {}".format(setting.pop("n_jobs"), method), INFO)

                    model = Learning(objective, method, xgb.XGBRegressor(**setting), cost_function)
                else:
                    log("1. Can't create model based on {}".format(method), ERROR)
            elif method.find("classifier") > -1:
                if method.find("calibration") > -1:
                    for key, value in setting.items():
                        if key.lower() not in ["base_estimator", "method", "cv"]:
                            del setting[key]

                    model = Learning(objective, method, CalibratedClassifierCV(**setting), cost_function)
                elif method.find("extratree") > -1:
                    model = Learning(objective, method, ExtraTreesClassifier(**setting), cost_function)
                elif method.find("randomforest") > -1:
                    model = Learning(objective, method, RandomForestClassifier(**setting), cost_function)
                elif method.find("gradientboosting") > -1:
                    model = Learning(objective, method, GradientBoostingClassifier(**setting), cost_function)
                elif method.find("xgboosting") > -1:
                    if "n_jobs" in setting:
                        log("Delete n_jobs={} from the setting for {}".format(setting.pop("n_jobs"), method), INFO)

                    model = Learning(objective, method, xgb.XGBClassifier(**setting), cost_function)
                else:
                   log("2. Can't create model based on {}".format(method), ERROR)
            else:
                log("3. Can't create model based on {}".format(method), ERROR)
        elif method.find("cluster") > -1:
            if method.find("kmeans") > -1:
                model = Learning(objective, method, KMeans(**setting))
            else:
                log("4. Can't create model based on {}".format(method), ERROR)
        elif method.find("deep") > -1:
            setting["folder"] = "{}/nn_layer={}_neurno={}_nfold={}_class_weight={}".format(setting["folder"], setting["number_of_layer"], setting["dimension"], setting["nfold"], setting["class_weight"][0])
            if not os.path.isdir(setting["folder"]):
                try:
                    os.makedirs(setting["folder"])
                except OSError as e:
                    pass

            log("The folder of deep learning is in {}".format(setting["folder"]), INFO)

            model = Learning(objective, method, None, cost_function)
            model.init_deep_params(**setting)
        elif method.find("customized") > -1:
            log("1. {}".format(setting), INFO)

            if "n_jobs" in setting:
                log("Delete the n_jobs={} from setting".format(setting.pop("n_jobs")), INFO)

            if "cost" in setting:
                log("Delete the cost={} from setting".format(setting.pop("cost")), INFO)

            setting["cost_func"] = cost_function

            if method.find("class") > -1:
                model = Learning(objective, method, CustomizedClassEstimator(**setting), cost_function)
            elif method.find("proba") > -1:
                model = Learning(objective, method, CustomizedProbaEstimator(**setting), cost_function)
            else:
                log("5. Can't create model based on {}".format(method), ERROR)
        else:
            log("6. Can't create model based on {}".format(method), ERROR)

        return model

class Learning(object):
    def __init__(self, objective, name, model, cost_function=log_loss):
        self.objective = objective
        self.name = name.lower()
        self.model = model

        if self.model:
            log("The parameters of {} are {}".format(self.name, self.model.get_params()), INFO)

        self.cost_function = cost_function

    def init_deep_params(self, nfold, folder, input_dims, learning_rate,
                         number_of_layer, batch_size, dimension,
                         nepoch, validation_split, class_weight,
                         callbacks=[]):

        self.batch_size = batch_size
        self.nepoch = nepoch

        self.callbacks = callbacks
        if self.callbacks:
            self.callbacks[-1].folder = folder

        self.class_weight = class_weight
        self.validation_split = validation_split

        self.model = logistic_regression(folder, number_of_layer, dimension, input_dims, learning_rate=learning_rate)

    def is_shallow_learning(self):
        return self.name.find("shallow") > -1

    def is_deep_learning(self):
        return self.name.find("deep") > -1

    def is_customized(self):
        return self.name.find("customized") > -1

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

    def preprocess_data(self, dataset):
        return dataset

    def train(self, train_x, train_y):
        train_x = self.preprocess_data(train_x)

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
            self.model.fit(train_x, train_y)
        elif self.is_deep_learning():
            self.model.fit(train_x, train_y, nb_epoch=self.nepoch, batch_size=self.batch_size, validation_split=self.validation_split, class_weight=self.class_weight, callbacks=self.callbacks)
        elif self.is_customized():
            self.model.fit(train_x, train_y)
        else:
            log("Not implement the training method of {}".format(self.name), ERROR)
            raise NotImplementError

    def predict(self, data):
        data = self.preprocess_data(data)

        if self.is_shallow_learning():
            if self.is_regressor():
                if self.name.find("logistic_regressor") > -1:
                    if self.objective == "binary_classification":
                        return self.model.predict_proba(data)[:, 1]
                    else:
                        return self.model.predict_proba(data)
                else:
                    return self.model.predict(data)
            elif self.is_classifier():
                # Only care the probability of class '1'
                if self.objective == "binary_classification":
                    return self.model.predict_proba(data)[:, 1]
                else:
                    return self.model.predict_proba(data)
            elif self.is_svm():
                if self.objective == "binary_classification":
                    return self.model.predict_proba(data)[:, 1]
                else:
                    return self.model.predict_proba(data)
        elif self.is_customized():
            if self.objective == "binary_classification":
                return self.model.predict_proba(data)[:, 1]
            else:
                return self.model.predict_proba(data)
        elif self.is_cluster():
            return self.model.predict(data)
        elif self.is_deep_learning():
            return [prob[0] if prob else 0.0 for prob in self.model.predict_proba(data)]
        else:
            log("Not implement the training method of {}".format(self.name), ERROR)
            raise NotImplementError

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
    def __init__(self, nfold):
        self.cost = {}
        self.nfold = nfold

    def insert_cost(self, model_name, nfold, cost):
        if model_name not in self.cost:
            self.cost.setdefault(model_name, np.zeros(self.nfold).astype(float))
            log("Not Found {} in self.cost, so creating it".format(model_name), INFO)
        else:
            log("Found {} in self.cost, so just inserting it".format(model_name), INFO)

        self.cost[model_name][nfold] = cost

class LearningQueue(object):
    def __init__(self, predictors, train_x, train_y, test_x, filepath=None):
        self.lock = threading.Lock()
        self.learning_queue = Queue.Queue()

        self.predictors = predictors
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.filepath = filepath

    def setup_layer_info(self, layer_two_training_dataset, layer_two_testing_dataset, learning_cost):
        self.layer_two_training_dataset = layer_two_training_dataset
        self.layer_two_testing_dataset = layer_two_testing_dataset
        self.learning_cost = learning_cost

    def put(self, nfold, model_idx, dataset_idxs, model):
        self.learning_queue.put((nfold, model_idx, dataset_idxs, model))

    def starts(self, models, objective, folder_model, folder_middle, cost_func, number_of_thread=1, saving_results=False):
        for idx in range(0, number_of_thread):
            worker = LearningThread(kwargs={"obj": self, "cost_func": cost_func, "models": models, "objective": objective, "model_folder": folder_model, "folder_middle": folder_middle, "saving_results": saving_results})
            worker.setDaemon(True)
            worker.start()

        self.learning_queue.join()

    def is_done_layer_two_training_dataset(self, layer_two_training_idx, model_idx):
        if np.sum(self.layer_two_training_dataset[layer_two_training_idx, model_idx] == 0.0) > 0.5*len(self.layer_two_training_dataset[layer_two_training_idx, model_idx]):
            return False
        else:
            return True

    def insert_layer_two_training_dataset(self, layer_two_training_idx, model_idx, results, params, filepath):
        self.lock.acquire()

        try:
            self.layer_two_training_dataset[layer_two_training_idx, model_idx] = results
            save_cache((params, layer_two_training_idx, model_idx, results), filepath)
        finally:
            self.lock.release()

    def insert_layer_two_testing_dataset(self, model_idx, nfold, results, params, filepath):
        self.lock.acquire()

        try:
            self.layer_two_testing_dataset[:, model_idx, nfold] = results
            save_cache((params, model_idx, nfold, results), filepath)
        finally:
            self.lock.release()

    def insert_cost(self, model_name, nfold, cost, filepath):
        self.learning_cost.insert_cost(model_name, nfold, cost)

        save_cache((model_name, nfold, cost), filepath)

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
            (nfold, model_idx, (train_x_idx, test_x_idx), p) = self.obj.learning_queue.get()
            timestamp_start = time.time()
            pair = copy.deepcopy(p)

            cost = -1
            model_name = pair[0]
            model_setting = pair[1]

            data_dimension, predictors = None, None
            if self.obj.predictors:
                if "data_dimension" in model_setting:
                    data_dimension = model_setting.pop("data_dimension")
                    predictors = sorted(self.obj.predictors[data_dimension])

                    log("Pop data_dimension from setting for {}".format(model_name), INFO)
                else:
                    log("Not found data_dimension from {}".format(model_name), INFO)

                log("{} gets {} features by {}".format(model_name, len(predictors), data_dimension), INFO)
            else:
                log("{} gets {} features by 'all'".format(model_name, len(self.obj.train_x[train_x_idx][0])), INFO)

            if model_name.find("deep") > -1:
                model_setting["folder"] = self.model_folder
                model_setting["nfold"] = nfold

            if "dependency" in model_setting:
                model_setting["base_estimator"] = LearningFactory.get_model(self.objective, model_setting.pop("dependency"), self.cost_func).model

            model_stamp = make_a_stamp(model_setting)
            feature_stamp = make_a_stamp(predictors)

            filepath_training = "{}/training_name={}_nfold={}_feature={}_setting={}.pkl".format(self.folder_middle, model_name, nfold, feature_stamp, model_stamp)
            filepath_testing = "{}/testing_name={}_nfold={}_feature={}_setting={}.pkl".format(self.folder_middle, model_name, nfold, feature_stamp, model_stamp)
            filepath_cost = "{}/cost_name={}_nfold={}_feature={}_setting={}.pkl".format(self.folder_middle, model_name, nfold, feature_stamp, model_stamp)

            log("The flag of self.saving_results is {}, and parent_folder is in {}".format(self.saving_results, os.path.dirname(filepath_training)), INFO)
            log("The filepath_training is {}({})".format(os.path.basename(filepath_training), os.path.exists(filepath_training)), INFO)
            log("The filepath_testing is {}({})".format(os.path.basename(filepath_testing), os.path.exists(filepath_testing)), INFO)
            log("The filepath_cost is {}({})".format(os.path.basename(filepath_cost), os.path.exists(filepath_cost)), INFO)

            train_x, train_y, validate_x, validate_y, test_x = None, self.obj.train_y[train_x_idx], None, self.obj.train_y[test_x_idx], None
            if self.obj.predictors and data_dimension:
                train_x = self.obj.train_x[predictors].values[train_x_idx]
                validate_x = self.obj.train_x[predictors].values[test_x_idx]
                test_x = self.obj.test_x[predictors]
            else:
                train_x = self.obj.train_x[train_x_idx]
                validate_x = self.obj.train_x[test_x_idx]
                test_x = self.obj.test_x

            if self.saving_results and os.path.exists(filepath_training) and os.path.exists(filepath_testing) and os.path.exists(filepath_cost):
                training_params, training_layer_two_training_idx, _, training_results = load_cache(filepath_training)
                testing_params, _, _, testing_results = load_cache(filepath_testing)
                _, _, cost = load_cache(filepath_cost)

                self.obj.insert_layer_two_training_dataset(training_layer_two_training_idx, model_idx, training_results, training_params, filepath_training)
                self.obj.insert_layer_two_testing_dataset(model_idx, nfold, testing_results, testing_params, filepath_testing)
                self.obj.insert_cost(model_name, nfold, cost, filepath_cost)

                log("Hit cache, the cost is {} based on {}'s {} nfold".format(cost, model_name, nfold), INFO)
            else:
                model = LearningFactory.get_model(self.objective, pair, self.cost_func)
                if model == None or model.model == None:
                    log("Can't init this model({})".format(model_name), WARN)
                elif model.is_cluster():
                    model.train(train_x, train_y)
                    training_results, testing_results = model.get_cluster_results(validate_x, test_x)

                    self.obj.insert_layer_two_training_dataset(test_x_idx, model_idx, training_results, model.model.get_params(), filepath_training)
                    self.obj.insert_layer_two_testing_dataset(model_idx, nfold, testing_results, model.model.get_params(), filepath_testing)
                    self.obj.insert_cost(model_name, nfold, -1, filepath_cost)
                else:
                    if model_name.find("deep") > -1 and nfold != 0:
                        # Copy the prediction results of training dataset to other folds
                        # All of them are already inserted when nfold equals zero

                        #Copy the prediction results of testing dataset to other folds
                        layer_two_testing_dataset = self.obj.layer_two_testing_dataset[:, model_idx, 0]
                        self.obj.insert_layer_two_testing_dataset(model.name, model_idx, nfold, layer_two_testing_dataset, {}, filepath_testing)

                        # Copy the cost of 0th fold into other folds
                        cost = self.obj.learning_cost[model_name][0]
                        self.obj.insert_cost(model_name, nfold, cost, filepath_cost)
                    else:
                        log("{} - {}, {}".format(model_name, train_x.shape, train_y.shape))
                        model.train(train_x, train_y)

                        results = model.predict(validate_x)
                        self.obj.insert_layer_two_training_dataset(test_x_idx, model_idx, results, model.model.get_params(), filepath_training)

                        layer_two_testing_dataset = model.predict(test_x)
                        self.obj.insert_layer_two_testing_dataset(model_idx, nfold, layer_two_testing_dataset, model.model.get_params(), filepath_testing)

                        cost = model.cost(validate_x, validate_y)
                        if np.isnan(cost):
                            log("The {} of '{}' model for {}th fold is NaN".format(self.cost_func.__name__, model_name, nfold), WARN)
                        else:
                            self.obj.insert_cost(model_name, nfold, cost, filepath_cost)

                        log("The grid score is {}".format(model.grid_scores()), DEBUG)

            timestamp_end = time.time()
            log("Cost {:02f} secends to train '{}' model for fold-{:02d}, and cost is {:.8f}".format(\
                    timestamp_end-timestamp_start, model_name, nfold, cost), INFO)

            self.obj.learning_queue.task_done()
            self.obj.dump()

def make_a_stamp(model_setting):
    m = md5.new()
    m.update(str(model_setting))
    return m.hexdigest()

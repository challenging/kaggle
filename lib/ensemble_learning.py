#!/usr/bin/env python

import os
import sys
import numpy as np
import pandas as pd

# For Shallow Learning
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.utils import shuffle

from learning import LearningFactory, Learning, LearningQueue, LearningCost
from customized_estimators import FinalEnsembleModel

from utils import log, INFO, ERROR
from load import load_cache, save_cache

def get_max_mean_min_probabilities(x):
    min_probabilities, max_probabilities, mean_probabilities = [], [], []
    for values in x:
        min_probabilities.append(np.nanmin(values))
        max_probabilities.append(np.nanmax(values))
        mean_probabilities.append(np.nanmean(values))

    return min_probabilities, max_probabilities, mean_probabilities

def store_layer_output(models, dataset, filepath, optional=[]):
    results_of_layer = {}
    for model_idx, model_name in enumerate(models):
        results_of_layer[model_name] = dataset[:, model_idx]

    if optional:
        for target in optional:
            results_of_layer.update(target)

    pd.DataFrame(results_of_layer).to_csv(filepath, index=False)

def get_learning_queue(models, n_folds, train_x, train_y, test_x, filepath_queue):
    layer_two_training_dataset = np.zeros((train_x.shape[0], len(models)))
    layer_two_testing_dataset = np.zeros((test_x.shape[0], len(models), n_folds))

    learning_cost, learning_queue = None, LearningQueue(train_x, train_y, test_x, filepath_queue)
    if filepath_queue and os.path.exists(filepath_queue):
        (layer_two_training_dataset, layer_two_testing_dataset, learning_cost) = load_cache(filepath_queue)
        learning_queue.setup_layer_info(layer_two_training_dataset, layer_two_testing_dataset, learning_cost)

        learning_cost = learning_queue.learning_cost
    else:
        learning_cost = LearningCost(n_folds)
        learning_queue.setup_layer_info(layer_two_training_dataset, layer_two_testing_dataset, learning_cost)

        learning_queue.dump()

    return learning_queue

def start_learning(objective, model_folder, train_x, train_y, test_x, models, n_folds, learning_queue, filepath_nfold, cost_func, number_of_thread=4, random_state=1201):
    skf = None
    if filepath_nfold and os.path.exists(filepath_nfold):
        skf = load_cache(filepath_nfold)

        log("Read skf from {}".format(filepath_nfold), INFO)
    else:
        if n_folds < 2:
            train_idx = shuffle([idx for idx in range(0, len(train_x))], random_state=1201)
            skf = [(train_idx, train_idx)]
        else:
            skf = list(StratifiedKFold(train_y, n_folds))
            save_cache(skf, filepath_nfold)

        log("Save skf({:2d} folds) in {}".format(n_folds, filepath_nfold), INFO)

    for model_idx, m in enumerate(models):
        model_name = m[0]
        for nfold, (train, test) in enumerate(skf):
            if learning_queue.is_done_layer_two_training_dataset(test, model_idx):
                log("fold-{:02d} data to '{}' model is done".format(nfold, model_name))
            else:
                if model_name.find("deep") == -1:
                    learning_queue.put(objective, model_folder, nfold, model_idx, (train, test), m)
                else:
                    if nfold == 0:
                        idxs = [idx for idx in range(0, len(train_x))]
                        learning_queue.put(objective, model_folder, nfold, model_idx, (idxs, idxs), m)
                    else:
                        continue

                log("Put fold-{:02d} data into this '{}' model".format(nfold, model_name))

    learning_queue.starts(cost_func, number_of_thread=number_of_thread)

    layer_two_testing_dataset = np.zeros((test_x.shape[0], len(models)))
    for idx in range(0, len(learning_queue.layer_two_testing_dataset)):
        layer_two_testing_dataset[idx] = learning_queue.layer_two_testing_dataset[idx].mean(axis=1)

    return layer_two_testing_dataset

def layer_model(objective, model_folder, train_x, train_y, test_x, models,
                filepath_queue, filepath_nfold,
                n_folds=10, cost_string="log_loss", number_of_thread=1,
                random_state=1201):

    cost_func = None
    if cost_string == "log_loss":
        cost_func = log_loss
    elif cost_string == "auc":
        cost_func = roc_auc_score

    number_of_feature = len(train_x[0])
    log("Data Distribution is ({}, {}), and then the number of feature is {}, and then prepare to save data in {}".format(np.sum(train_y==0), np.sum(train_y==1), number_of_feature, model_folder), INFO)

    learning_queue = get_learning_queue(models, n_folds, train_x, train_y, test_x, filepath_queue)
    layer_two_testing_dataset = start_learning(objective, model_folder, train_x, train_y, test_x, models, n_folds, learning_queue, filepath_nfold, cost_func, number_of_thread, random_state)

    return learning_queue.layer_two_training_dataset, layer_two_testing_dataset, learning_queue.learning_cost

def final_model(pair, train_x, train_y, test_x, cost_string="log_loss"):
    log("The phase 3 starts...", INFO)

    cost_function = None
    if cost_string == "log_loss":
        cost_function = log_loss
    elif cost_string == "auc":
        cost_function = roc_auc_score
    else:
        log("Please set the cost function", ERROR)
        sys.exit(1)

    model = LearningFactory.get_model(pair)
    model.train(train_x, train_y)

    log("The weights of {} are {}".format(model.name, model.coef()), INFO)

    prediction_results_training = model.predict(train_x)
    log("The cost of layer-3 model is {}".format(cost_function(train_y, prediction_results_training)))

    return model.predict(test_x)

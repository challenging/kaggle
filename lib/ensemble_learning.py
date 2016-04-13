#!/usr/bin/env python

import os
import sys
import numpy as np
import pandas as pd

# For Shallow Learning
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.utils import shuffle

from learning import LearningFactory, Learning, LearningQueue, LearningCost
from utils import log, INFO, ERROR
from load import load_cache, save_cache

def get_max_mean_min_probabilities(x):
    min_probabilities, max_probabilities, mean_probabilities = [], [], []
    for values in x:
        min_probabilities.append(np.nanmin(values))
        max_probabilities.append(np.nanmax(values))
        mean_probabilities.append(np.nanmean(values))

    return min_probabilities, max_probabilities, mean_probabilities

def store_layer_output(models, dataset, filepath, targets=[]):
    results_of_layer = {}
    for model_idx, model_name in enumerate(models):
        results_of_layer.setdefault(model_name, [])

        for idx in range(0, len(dataset)):
            results_of_layer[model_name].append(dataset[idx][model_idx])

    if targets:
        for target in targets:
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
        only_models = [model[0] for model in models]

        learning_cost = LearningCost(only_models, n_folds)
        learning_queue.setup_layer_info(layer_two_training_dataset, layer_two_testing_dataset, learning_cost)

        learning_queue.dump()

    return learning_queue

def start_learning(model_folder, train_x, train_y, test_x, models, n_folds, learning_queue, filepath_nfold, number_of_thread=4, random_state=1201):
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
        for nfold, (train, test) in enumerate(skf):
            model_name = m[0]
            if learning_queue.is_done_layer_two_training_dataset(test, model_idx):
                log("fold-{:02d} data to '{}' model is done".format(nfold, model_name))
            else:
                learning_queue.put(model_folder, nfold, model_idx, (train, test), m)
                log("Put fold-{:02d} data into this '{}' model".format(nfold, model_name))

    learning_queue.starts(number_of_thread=number_of_thread)

    layer_two_testing_dataset = np.zeros((test_x.shape[0], len(models)))
    for idx in range(0, len(learning_queue.layer_two_testing_dataset)):
        layer_two_testing_dataset[idx] = learning_queue.layer_two_testing_dataset[idx].mean(axis=1)

    return layer_two_testing_dataset

def layer_one_model(model_folder, train_x, train_y, test_x, models,
                    filepath_training, filepath_queue, filepath_nfold,
                    n_folds=10, number_of_thread=1,
                    random_state=1201):

    log("The phase 1 starts...", INFO)

    number_of_feature = len(train_x[0])
    log("Data Distribution is ({}, {}), and then the number of feature is {}, and then prepare to save data in {}".format(np.sum(train_y==0), np.sum(train_y==1), number_of_feature, model_folder), INFO)

    learning_queue = get_learning_queue(models, n_folds, train_x, train_y, test_x, filepath_queue)
    layer_two_testing_dataset = start_learning(model_folder, train_x, train_y, test_x, models, n_folds, learning_queue, filepath_nfold, number_of_thread, random_state)

    # Save the results of layer1
    targets = [{"Target": train_y}]
    store_layer_output([m[0] for m in models], learning_queue.layer_two_training_dataset, filepath_training, targets=targets)

    return learning_queue.layer_two_training_dataset, layer_two_testing_dataset, learning_queue.learning_cost

def layer_two_model(model_folder, train_x, train_y, test_x, models,
                    filepath_training, filepath_queue, filepath_nfold,
                    n_folds=10, number_of_thread=1,
                    random_state=1201):

    log("The phase 2 starts...")

    zero_train_x = 1 - train_x
    train_X = np.insert(train_x, range(0, len(train_x[0])), zero_train_x, axis=1)

    zero_test_x = 1 - test_x
    test_X = np.insert(test_x, range(0, len(test_x[0])), zero_test_x, axis=1)

    number_of_feature = len(train_X[0])
    log("Data Distribution is ({}, {}), and then the number of feature is {}, and then prepare to save data in {}".format(np.sum(train_y==0), np.sum(train_y==1), number_of_feature, model_folder), INFO)

    learning_queue = get_learning_queue(models, n_folds, train_X, train_y, test_X, filepath_queue)
    layer_two_testing_dataset = start_learning(model_folder, train_X, train_y, test_x, models, n_folds, learning_queue, filepath_nfold, number_of_thread, random_state)

    targets = [{"Target": train_y}]
    store_layer_output([m[0] for m in models], train_X, filepath_training, targets=targets)

    return learning_queue.layer_two_training_dataset, layer_two_testing_dataset, learning_queue.learning_cost

def layer_three_model(train_x, train_y, test_x, cost_string="logloss"):
    log("The phase 3 starts...", INFO)

    cost_function = None
    if cost_string == "logloss":
        cost_function = log_loss
    elif cost_string == "auc":
        cost_function = auc
    else:
        log("Please set the cost function", ERROR)
        sys.exit(1)

    prediction_results_training = np.zeros((len(train_x)))
    prediction_results_training[:] += prediction_results_training[:, 0]*4./9 + prediction_results_training[:, 1]*5./9

    log("The cost of layer-3 model is {}".format(cost_function(train_y, prediction_results_training)))

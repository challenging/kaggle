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
from utils import log, INFO
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

def layer_one_model(model_folder, train_x, train_y, test_x, test_id, models, layer2_model_name,
                    cost_string="logloss", n_folds=10, number_of_thread=1, filepath_queue=None, filepath_nfold=None):

    layer_two_training_dataset = np.zeros((train_x.shape[0], len(models)))
    layer_two_testing_dataset = np.zeros((test_x.shape[0], len(models), n_folds))

    learning_cost, learning_queue = None, LearningQueue(train_x, train_y, test_x, filepath_queue)
    if filepath_queue and os.path.exists(filepath_queue):
        (layer_two_training_dataset, layer_two_testing_dataset, learning_cost) = load_cache(filepath_queue)
        learning_queue.setup_layer_info(layer_two_training_dataset, layer_two_testing_dataset, learning_cost)

        learning_cost = learning_queue.learning_cost
    else:
        only_models = [model[0] for model in models]
        only_models += [layer2_model_name]

        learning_cost = LearningCost(only_models, n_folds)
        learning_queue.setup_layer_info(layer_two_training_dataset, layer_two_testing_dataset, learning_cost)

        learning_queue.dump()

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

    return learning_queue.layer_two_training_dataset, layer_two_testing_dataset, learning_cost

def layer_two_model(models, train_x, train_y, test_id, test_x, learning_cost, model_name, filepath_training, filepath_testing, filepath_cost,
                    cost_string="logloss"):

    cost_function = None
    if cost_string == "logloss":
        cost_function = log_loss
    elif cost_string == "auc":
        cost_function = auc

    only_models = [model[0] for model in models]

    model = LearningFactory.get_model(model_name)

    model.train(train_x, train_y)
    training_prediction_results = model.predict(train_x)

    log("The decision function is {}".format(model.coef()), INFO)

    min_probabilities, max_probabilities, mean_probabilities = get_max_mean_min_probabilities(train_x)
    cost_min = cost_function(train_y, min_probabilities)
    cost_mean = cost_function(train_y, mean_probabilities)
    cost_max = cost_function(train_y, max_probabilities)

    # If use average, the cost will be ...
    log("The cost of min/mean/max-model is {:.8f}/{:.8f}/{:.8f}".format(cost_min, cost_mean, cost_max), INFO)

    targets = []
    for key, values in {"Target": train_y, "Probability of Layer 2": training_prediction_results, "min.": min_probabilities, "max.": max_probabilities, "avg.": mean_probabilities}.items():
        targets.append({key: values})

    store_layer_output(only_models, train_x, filepath_training, targets=targets)

    min_probabilities, max_probabilities, mean_probabilities = get_max_mean_min_probabilities(test_x)
    targets = []
    for key, values in {"ID":test_id ,"min.": min_probabilities, "max.": max_probabilities, "avg.": mean_probabilities}.items():
        targets.append({key: values})

    store_layer_output(only_models, test_x, filepath_testing, targets=targets)

    cost = cost_function(train_y, training_prediction_results)

    max_proba = np.max(training_prediction_results, axis=0)
    norm_training_prediction_results = training_prediction_results / max_proba
    norm_cost = cost_function(train_y, norm_training_prediction_results)

    log("The cost of layer2 model is {:.8f}/{:8f}".format(cost, norm_cost), INFO)

    # Hardcode to set nfold to be ZERO, and then save it
    learning_cost.insert_cost(model.name, 0, cost)
    save_cache(learning_cost, filepath_cost)

    return model.predict(test_x)

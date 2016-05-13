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

def get_learning_queue(predictors, models, n_folds, train_x, train_y, test_x, filepath_queue):
    layer_two_training_dataset = np.zeros((train_x.shape[0], len(models)))
    layer_two_testing_dataset = np.zeros((test_x.shape[0], len(models), n_folds))

    learning_cost, learning_queue = None, LearningQueue(predictors, train_x, train_y, test_x, filepath_queue)
    if filepath_queue and os.path.exists(filepath_queue):
        (layer_two_training_dataset, layer_two_testing_dataset, learning_cost) = load_cache(filepath_queue)
        learning_queue.setup_layer_info(layer_two_training_dataset, layer_two_testing_dataset, learning_cost)

        learning_cost = learning_queue.learning_cost
    else:
        learning_cost = LearningCost(n_folds)
        learning_queue.setup_layer_info(layer_two_training_dataset, layer_two_testing_dataset, learning_cost)

        learning_queue.dump()

    return learning_queue

def start_learning(objective, folder_model, folder_middle,
                   train_x, train_y, test_x, models, n_folds, learning_queue, filepath_nfold, cost_func,
                   number_of_thread=4, random_state=1201, saving_results=False):
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
            #if model_name.find("deep") == -1:
            learning_queue.put(nfold, model_idx, (train, test), m)
            #else:
            #    if nfold == 0:
            #        idxs = [idx for idx in range(0, len(train_x))]
            #        learning_queue.put(nfold, model_idx, (idxs, idxs), m)
            #    else:
            #        continue

            log("Put fold-{:02d} data into this '{}' model".format(nfold, model_name))

    learning_queue.starts(models, objective, folder_model, folder_middle, cost_func, number_of_thread=number_of_thread, saving_results=saving_results)

    layer_two_testing_dataset = np.zeros((test_x.shape[0], len(models)))
    for idx in range(0, len(learning_queue.layer_two_testing_dataset)):
        layer_two_testing_dataset[idx] = learning_queue.layer_two_testing_dataset[idx].mean(axis=1)

    return layer_two_testing_dataset

def layer_model(objective, folder_model, folder_middle, predictors, train_x, train_y, test_x, models, filepath_queue, filepath_nfold,
                n_folds=10, cost_string="log_loss", number_of_thread=1, random_state=1201, saving_results=False):

    cost_func = None
    if cost_string == "auc":
        cost_func = roc_auc_score
    else:
        cost_func = log_loss
    log("The cost function is {}".format(cost_func.__name__), INFO)

    log("Data Distribution is ({}, {}), and then prepare to save data in {}, and the saving_results is {}".format(\
            np.sum(train_y==0), np.sum(train_y==1), folder_model, saving_results), INFO)

    learning_queue = get_learning_queue(predictors, models, n_folds, train_x, train_y, test_x, filepath_queue)
    layer_two_testing_dataset = start_learning(objective, folder_model, folder_middle,
                                               train_x, train_y, test_x, models, n_folds, learning_queue, filepath_nfold, cost_func,
                                               number_of_thread, random_state, saving_results)

    return learning_queue.layer_two_training_dataset, layer_two_testing_dataset, learning_queue.learning_cost

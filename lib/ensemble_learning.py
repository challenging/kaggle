#!/usr/bin/env python

import os
import sys
import pickle
import numpy as np
import pandas as pd

# For Shadow Learning
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss

from learning import LearningFactory, Learning, LearningQueue, LearningLogLoss
from utils import log, INFO

def store_layer_output(models, dataset, filepath, target=[], ids=[]):
    results_of_layer = {"ID": []}
    for model_idx, model_name in enumerate(models):
        results_of_layer.setdefault(model_name, [])

        for idx in range(0, len(dataset)):
            if model_idx == 0:
                if np.any(ids):
                    results_of_layer["ID"].append(ids[idx])
                else:
                    results_of_layer["ID"].append(idx)

            results_of_layer[model_name].append(dataset[idx][model_idx])

    if np.any(target):
        results_of_layer.update(target)

    pd.DataFrame(results_of_layer).to_csv(filepath, index=False)

def layer_one_model(model_folder, train_x, train_y, test_x, test_id, models, layer2_model_name, n_folds=10, number_of_thread=1, filepath_queue=None, filepath_nfold=None):
    skf = list(StratifiedKFold(train_y, n_folds))

    layer_two_training_dataset = np.zeros((train_x.shape[0], len(models)))
    layer_two_testing_dataset = np.zeros((test_x.shape[0], len(models), n_folds))

    learning_logloss, learning_queue = None, LearningQueue(train_x, train_y, test_x, filepath_queue)
    if filepath_queue and os.path.exists(filepath_queue):
        with open(filepath_queue, "rb") as INPUT:
            (layer_two_training_dataset, layer_two_testing_dataset, learning_logloss) = pickle.load(INPUT)
            learning_queue.setup_layer_info(layer_two_training_dataset, layer_two_testing_dataset, learning_logloss)

            learning_logloss = learning_queue.learning_logloss

        if "layer2_lr" in learning_logloss.logloss:
            learning_logloss.logloss[layer2_model_name] = learning_logloss.logloss["layer2_lr"]
            del learning_logloss.logloss["layer2_lr"]
    else:
        learning_logloss = LearningLogLoss(models + [layer2_model_name], n_folds)
        learning_queue.setup_layer_info(layer_two_training_dataset, layer_two_testing_dataset, learning_logloss)

        learning_queue.dump()

    skf = None
    if filepath_nfold and os.path.exists(filepath_nfold):
        with open(filepath_nfold, "rb") as INPUT:
            skf = pickle.load(INPUT)

        log("Read skf from {}".format(filepath_nfold), INFO)
    else:
        skf = list(StratifiedKFold(train_y, n_folds))
        with open(filepath_nfold, "wb") as OUTPUT:
            pickle.dump(skf, OUTPUT)

        log("Save skf in {}".format(filepath_nfold), INFO)

    for nfold, (train, test) in enumerate(skf):
        for model_idx, model_name in enumerate(models):
            if learning_queue.is_done_layer_two_training_dataset(test, model_idx):
                log("fold-{:02d} data to '{}' model is done".format(nfold, model_name))
            else:
                learning_queue.put(nfold, model_idx, (train, test), model_name)
                log("Put fold-{:02d} data into this '{}' model".format(nfold, model_name))

    learning_queue.starts(number_of_thread=number_of_thread)

    layer_two_testing_dataset = np.zeros((test_x.shape[0], len(models)))
    for idx in range(0, len(learning_queue.layer_two_testing_dataset)):
        layer_two_testing_dataset[idx] = learning_queue.layer_two_testing_dataset[idx].mean(axis=1)

    # Save testing output of the layer one
    filepath_first_layer_dataset = "{}/testing_layer1.csv".format(model_folder)
    store_layer_output(models, layer_two_testing_dataset, filepath_first_layer_dataset, ids=test_id)

    return learning_queue.layer_two_training_dataset, layer_two_testing_dataset, learning_logloss

def layer_two_model(layer_one_models, train_x, train_y, test_x, learning_logloss, model_name, model_folder, deep_setting={}):
    model = LearningFactory.get_model(model_name)
    if deep_setting:
        model.init_deep_params(model_folder, deep_setting["number_of_layer"], deep_setting["mini_batch"],
                               deep_setting["dimension"], train_x, train_y, len(train_x[0]),
                               deep_setting["nepoch"], deep_setting["callbacks"])

    model.train(train_x, train_y)
    log("The decision function is {}".format(model.coef()), INFO)

    # Save the training output for layer 1/2
    training_prediction_results = model.predict(train_x)

    # min/max/mean probability calculation
    min_probabilities, max_probability, mean_probability = [], [], []
    for values in train_x:
        min_probabilities.append(np.nanmin(values))
        max_probabilities.append(np.nanmax(values))
        mean_probabilities.append(np.nanmean(values))

    filepath_layer_dataset = "{}/training_layer.csv".format(model_folder)
    results = {"Target": train_y, "Probability of Layer 2": training_prediction_results, "min.": min_probabilities, "max.": max_probability, "avg.": mean_probability}
    store_layer_output(layer_one_models, train_x, filepath_layer_dataset, target=results)

    cost = log_loss(train_y, training_prediction_results)
    log("The overall logloss is {:.8f}".format(cost))

    # Hardcode to set nfold to be ZERO, and then save it
    learning_logloss.insert_logloss(model.name, 0, cost)

    with open("{}/logloss.pickle".format(model_folder), "wb") as OUTPUT:
        pickle.dump(learning_logloss, OUTPUT)

    return model.predict(test_x)

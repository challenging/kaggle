#!/usr/bin/env python

import os
import sys
import pickle
import numpy as np
import pandas as pd

# For Shallow Learning
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.utils import shuffle

from learning import LearningFactory, Learning, LearningQueue, LearningLogLoss
from utils import log, INFO

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
                    n_folds=10, number_of_thread=1, filepath_queue=None, filepath_nfold=None):
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

        for model_name in learning_logloss.logloss.keys():
            if model_name.find("shadow") > -1:
                learning_logloss.logloss[model_name.replace("shadow", "shallow")] = learning_logloss.logloss[model_name]
                del learning_logloss.logloss[model_name] 
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
        if n_folds < 2:
            train_idx = shuffle([idx for idx in range(0, len(train_x))], random_state=1201)
            skf = [(train_idx, train_idx)]
        else:
            skf = list(StratifiedKFold(train_y, n_folds))
            with open(filepath_nfold, "wb") as OUTPUT:
                pickle.dump(skf, OUTPUT)

        log("Save skf({:2d} folds) in {}".format(n_folds, filepath_nfold), INFO)

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

    return learning_queue.layer_two_training_dataset, layer_two_testing_dataset, learning_logloss

def layer_two_model(layer_one_models, train_x, train_y, test_id, test_x, learning_logloss, model_name, filepath_training, filepath_testing, filepath_logloss,
                    deep_setting={}):
    model = LearningFactory.get_model(model_name)
    if deep_setting:
        model.init_deep_params(deep_setting["folder_weights"],
                               len(train_x[0]),
                               deep_setting["number_of_layer"],
                               deep_setting["batch_size"],
                               deep_setting["dimension"],
                               deep_setting["nepoch"],
                               deep_setting.get("validation_split", 0.15),
                               deep_setting.get("class_weight", None),
                               deep_setting["callbacks"])

        model.train([train_x, train_x], train_y)
        training_prediction_results = model.predict([train_x, train_x])
    else:
        model.train(train_x, train_y)
        training_prediction_results = model.predict(train_x)

    log("The decision function is {}".format(model.coef()), INFO)

    # min/max/mean probability calculation
    min_probabilities, max_probabilities, mean_probabilities = [], [], []
    for values in train_x:
        min_probabilities.append(np.nanmin(values))
        max_probabilities.append(np.nanmax(values))
        mean_probabilities.append(np.nanmean(values))

    cost_min = log_loss(train_y, min_probabilities)
    cost_mean = log_loss(train_y, mean_probabilities)
    cost_max = log_loss(train_y, max_probabilities)

    targets = []
    for key, values in {"Target": train_y, "Probability of Layer 2": training_prediction_results, "min.": min_probabilities, "max.": max_probabilities, "avg.": mean_probabilities}.items():
        targets.append({key: values})
    store_layer_output(layer_one_models, train_x, filepath_training, targets=targets)

    # Save the testing output for layer 1
    min_probabilities, max_probabilities, mean_probabilities = [], [], []
    for values in test_x:
        min_probabilities.append(np.nanmin(values))
        max_probabilities.append(np.nanmax(values))
        mean_probabilities.append(np.nanmean(values))

    targets = []
    for key, values in {"ID":test_id ,"min.": min_probabilities, "max.": max_probabilities, "avg.": mean_probabilities}.items():
        targets.append({key: values})
    store_layer_output(layer_one_models, test_x, filepath_testing, targets=targets)

    cost = log_loss(train_y, training_prediction_results)
    log("The logloss of layer2 model is {:.8f}".format(cost), INFO)

    # If use average, the logloss will be ...
    log("The logloss of min/mean/max-model is {:.8f}/{:.8f}/{:.8f}".format(cost_min, cost_mean, cost_max), INFO)

    # Hardcode to set nfold to be ZERO, and then save it
    learning_logloss.insert_logloss(model.name, 0, cost)

    with open(filepath_logloss, "wb") as OUTPUT:
        pickle.dump(learning_logloss, OUTPUT)

    return model.predict(test_x)

#!/usr/bin/env python

import os
import sys
import pickle
import numpy as np
import pandas as pd

# For Shadow Learning
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import log_loss

from learning import LearningQueue, LearningLogLoss
from utils import log, INFO

def ensemble_model(model_folder, train_x, train_y, test_x, test_id, models, n_folds=10, number_of_thread=1, filepath_queue=None, filepath_nfold=None):
    skf = list(StratifiedKFold(train_y, n_folds))

    layer_two_training_dataset = np.zeros((train_x.shape[0], len(models)))
    layer_two_testing_dataset = np.zeros((test_x.shape[0], len(models), n_folds))

    learning_logloss, learning_queue = None, LearningQueue(train_x, train_y, test_x, filepath_queue)
    if filepath_queue and os.path.exists(filepath_queue):
        with open(filepath_queue, "rb") as INPUT:
            (layer_two_training_dataset, layer_two_testing_dataset, learning_logloss) = pickle.load(INPUT)
            learning_queue.setup_layer_info(layer_two_training_dataset, layer_two_testing_dataset, learning_logloss)

            learning_logloss = learning_queue.learning_logloss
    else:
        learning_logloss = LearningLogLoss(models + ["layer2_lr"], n_folds)
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

    # Save the output of 1-layer
    def store_layer_output(dataset, filepath, target=[], ids=[]):
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
            results_of_layer["Probability of Layer 2"] = target["Probability of Layer 2"]
            results_of_layer["Target"] = target["Target"]

        pd.DataFrame(results_of_layer).to_csv(filepath, index=False)

    # Save testing output of the layer one
    filepath_first_layer_dataset = "{}/testing_layer1.csv".format(model_folder)
    store_layer_output(layer_two_testing_dataset, filepath_first_layer_dataset, ids=test_id)

    layer_2_model = LogisticRegression()
    params = {"C": [1e-04, 1e-02, 1e-01, 1, 1e+01, 1e+02, 1e+04],
              "solver": ["newton-cg", "lbfgs", "liblinear"]}
    clf = GridSearchCV(layer_2_model, params, verbose=1)
    clf.fit(learning_queue.layer_two_training_dataset, train_y)
    log("The decision function is {}".format(clf.best_estimator_.coef_), INFO)

    # Save the training output for layer 1/2
    training_prediction_results = clf.predict_proba(learning_queue.layer_two_training_dataset)[:,1]
    filepath_layer_dataset = "{}/training_layer.csv".format(model_folder)
    results = {"Target": [], "Probability of Layer 2": training_prediction_results}
    for value in train_y:
        results["Target"].append(value)
    store_layer_output(layer_two_training_dataset, filepath_layer_dataset, target=results)

    cost = log_loss(train_y, clf.predict_proba(learning_queue.layer_two_training_dataset)[:,1])

    # Hardcode to set nfold to be ZERO, and then save it
    learning_logloss.insert_logloss("layer2_lr", 0, cost)

    with open("{}/logloss.pickle".format(model_folder), "wb") as OUTPUT:
        pickle.dump(learning_logloss, OUTPUT)

    log("The overall logloss is {:.8f}".format(cost))

    return clf.predict_proba(layer_two_testing_dataset)[:,1]

#!/usr/bin/env python

import os
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss, roc_auc_score

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Merge
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint

from utils import log, DEBUG, INFO, ERROR

class KaggleCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, save_best_only=True, training_set=(None, None), testing_set=(None, None), folder=None, cost_string="log_loss", save_training_dataset=False, verbose=1):
        ModelCheckpoint.__init__(self, filepath=filepath, save_best_only=save_best_only, verbose=1)

        self.training_x, self.training_y = training_set
        self.testing_x, self.testing_id, = testing_set
        self.folder = folder
        self.save_training_dataset = save_training_dataset

        if cost_string == "log_loss":
            self.cost_function = cost_string
        elif cost_string == "auc":
            self.cost_function = roc_auc_score
        else:
            log("Found undefined cost function - {}".format(cost_string), ERROR)
            raise NotImplementError

    def save_results(self, filepath, proba, is_testing=False):
        probas = [prob[0] if prob[0] else 0.0 for prob in proba]

        results = {"PredictedProb": probas}
        if is_testing:
            results["ID"] = self.testing_id
        else:
            results["Target"] = self.training_y

        pd.DataFrame(results).to_csv(filepath, index=False)
        log("Save the results in {}".format(filepath), DEBUG)

    def on_epoch_end(self, epoch, logs={}):
        filepath = self.folder + "/" + self.filepath.format(epoch=epoch+1, **logs)

        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can save best model only with %s available, skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    log("Epoch %05d: %s improved from %0.8f to %0.8f" % (epoch+1, self.monitor, self.best, current), INFO)

                    self.best = current
                    self.model.save_weights(filepath, overwrite=True)

                    # Save the prediction results for testing set
                    if self.save_training_dataset and self.folder:
                        # Save the training results
                        proba_training = self.model.predict_proba(self.training_x)

                        filepath_training = "{}/training_{:05d}.csv".format(self.folder, epoch+1)
                        self.save_results(filepath_training, proba_training)

                        cost = self.cost_function(self.training_y, proba_training)
                        log("Epoch {:05d}: current {} is {:.8f}".format(epoch+1, self.cost_function.__name__, cost), INFO)
                else:
                    if self.verbose > 0:
                        log('Epoch %05d: %s did not improve' %(epoch, self.monitor), DEBUG)
        else:
            if self.verbose > 0:
                log('Epoch %05d: saving model to %s' % (epoch+1, filepath), DEBUG)

            self.model.save_weights(filepath, overwrite=True)

def get_newest_model(folder):
    import glob

    newest = None
    try:
        newest = max(glob.iglob("{}/*.weights.hdf5".format(folder)), key=lambda x: os.path.getmtime(x))
    except:
        pass

    return newest

def logistic_regression(model_folder, layer, dimension, number_of_feature,
       cost="binary_crossentropy", learning_rate=1e-6, dropout_rate=0.5, nepoch=10, activation="tanh"):

    model = Sequential()
    model.add(Dense(dimension, input_dim=number_of_feature, init="uniform", activation=activation))
    model.add(Dropout(dropout_rate))

    for idx in range(0, layer-2, 1):
        model.add(Dense(dimension, input_dim=dimension, init="uniform", activation=activation))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, init="uniform", activation="sigmoid"))

    optimizer = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-06)
    model.compile(loss=cost, optimizer=optimizer, metrics=['accuracy'])

    filepath_model = get_newest_model(model_folder)
    if filepath_model:
        model.load_weights(filepath_model)

        log("Load weights from {}".format(filepath_model), INFO)
    else:
        log("A new one model, {}".format(model_folder), INFO)

    return model

def logistic_regression_2(model_folder, layer, dimension, input_dims,
       cost="binary_crossentropy", learning_rate=1e-6, dropout_rate=0.5, nepoch=10, init="uniform", activation="tanh"):

    sources = []
    for input_dim in input_dims:
        model = Sequential()
        model.add(Dense(dimension, input_dim=input_dim, init=init, activation=activation))
        model.add(Dropout(dropout_rate))

        sources.append(model)

    sources_dot = []
    for idx in range(0, len(sources), 2):
        model = Sequential()
        model.add(Merge(sources[idx: idx+2], mode="dot"))
        model.add(Dense(dimension, input_dim=dimension, init=init, activation=activation))

        sources_dot.append(model)

    model = Sequential()
    model.add(Merge(sources_dot, mode="sum"))
    model.add(Dense(dimension, input_dim=dimension, init=init, activation=activation))
    model.add(Dropout(dropout_rate))

    for idx in range(0, layer-3, 1):
        model.add(Dense(dimension, input_dim=dimension, init=init, activation=activation))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, init=init, activation="softmax"))

    optimizer = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-06)
    model.compile(loss=cost, optimizer=optimizer, metrics=['accuracy'])

    filepath_model = get_newest_model(model_folder)
    if filepath_model:
        model.load_weights(filepath_model)

        log("Load weights from {}".format(filepath_model), INFO)
    else:
        log("A new one model, {}".format(model_folder), INFO)

    return model

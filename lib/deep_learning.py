#!/usr/bin/env python

import os
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Merge
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint

from utils import log, DEBUG, INFO

class KaggleCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, save_best_only=True, training_set=(None, None), testing_set=(None, None), folder=None, verbose=1):
        ModelCheckpoint.__init__(self, filepath=filepath, save_best_only=save_best_only, verbose=1)

        self.training_x, self.training_y = training_set
        self.testing_x, self.testing_id, = testing_set
        self.folder = folder

    def save_results(self, filepath, proba, base_proba=None, is_testing=False):
        probas = [prob[0] if prob[0] else 0.0 for prob in proba]

        results = {"PredictedProb": probas}
        if is_testing:
            results["ID"] = self.testing_id
        else:
            results["Target"] = self.training_y

        pd.DataFrame(results).to_csv(filepath, index=False)
        log("Save the results in {}".format(filepath), DEBUG)

        if not base_proba:
            base_proba = np.max(results["PredictedProb"], axis=0)

        results["PredictedProb"] = results["PredictedProb"] / base_proba

        filepath_normalization = filepath.replace(".csv", "_normalization.csv")
        pd.DataFrame(results).to_csv(filepath_normalization, index=False)
        log("Save the normalized results in {}".format(filepath_normalization), DEBUG)

        return base_proba

    def on_epoch_end(self, epoch, logs={}):
        filepath = self.filepath.format(epoch=epoch+1, **logs)
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can save best model only with %s available, skipping.' % (self.monitor), RuntimeWarning)
            else:
                proba_training = self.model.predict_proba(self.training_x)

                max_proba = np.max(proba_training, axis=0)
                norm_proba_training = proba_training / max_proba

                if self.monitor_op(current, self.best):
                    log("Epoch %05d: %s improved from %0.8f to %0.8f" % (epoch+1, self.monitor, self.best, current), INFO)

                    self.best = current
                    self.model.save_weights(filepath, overwrite=True)

                    # Save the prediction results for testing set
                    if self.folder:
                        # Save the training results
                        filepath_training = "{}/training_{:05d}.csv".format(self.folder, epoch+1)
                        base_proba = self.save_results(filepath_training, proba_training)

                        # Save the testing results
                        filepath_testing = "{}/submission_{:05d}.csv".format(self.folder, epoch+1)
                        proba = self.model.predict_proba(self.testing_x)
                        self.save_results(filepath_testing, proba, base_proba, is_testing=True)
                else:
                    if self.verbose > 0:
                        log('Epoch %05d: %s did not improve' %(epoch, self.monitor), DEBUG)

                logloss = log_loss(self.training_y, proba_training)
                norm_logloss = log_loss(self.training_y, norm_proba_training)
                log("Epoch {:05d}: current logloss is {:.8f}/{:.8f}".format(epoch+1, logloss, norm_logloss), INFO)
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

def logistic_regression(model_folder, layer, batch_size, dimension, number_of_feature,
       learning_rate=1e-6, dropout_rate=0.5, nepoch=10, activate_function="sigmoid"):

    model = Sequential()
    model.add(Dense(dimension, input_dim=number_of_feature, init="uniform"))
    model.add(Activation(activate_function))
    model.add(Dropout(dropout_rate))

    for idx in range(0, layer-2, 1):
        model.add(Dense(dimension, input_dim=dimension, init="uniform"))
        model.add(Activation(activate_function))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, init="uniform"))
    model.add(Activation("sigmoid"))

    optimizer = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-06)
    model.compile(loss="binary_crossentropy", optimizer=optimizer)

    filepath_model = get_newest_model(model_folder)
    if filepath_model:
        model.load_weights(filepath_model)

        log("Load weights from {}".format(filepath_model), INFO)
    else:
        log("A new one model, {}".format(model_folder), INFO)

    return model


def logistic_regression_2(model_folder, layer, batch_size, dimension, input_dims,
       learning_rate=1e-6, dropout_rate=0.5, nepoch=10, activation="tanh"):

    sources = []
    for input_dim in input_dims:
        model_a = Sequential()
        model_a.add(Dense(dimension, input_dim=input_dim, init="uniform", activation=activation))
        model_a.add(Dropout(dropout_rate))

        sources.append(model_a)

    model_a = Sequential()
    model_a.add(Merge(sources[:2], mode="dot"))

    model_b = Sequential()
    model_b.add(Merge(sources[2:], mode="dot"))

    model = Sequential()
    model.add(Merge([model_a, model_b], mode="sum"))
    model.add(Dense(dimension, input_dim=dimension, init="uniform", activation=activation))
    model.add(Dropout(dropout_rate))

    for idx in range(0, layer-3, 1):
        model.add(Dense(dimension, input_dim=dimension, init="uniform", activation=activation))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, init="uniform", activation=activation))

    optimizer = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-06)
    model.compile(loss="binary_crossentropy", optimizer=optimizer)

    filepath_model = get_newest_model(model_folder)
    if filepath_model:
        model.load_weights(filepath_model)

        log("Load weights from {}".format(filepath_model), INFO)
    else:
        log("A new one model, {}".format(model_folder), INFO)

    return model

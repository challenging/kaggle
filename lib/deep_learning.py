#!/usr/bin/env python

import os
import sys

import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint

class KaggleCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, save_best_only=True, testing_set=(None, None), folder_testing=None, verbose=1):
        ModelCheckpoint.__init__(self, filepath=filepath, save_best_only=save_best_only, verbose=1)

        self.testing_x, self.testing_id, = testing_set
        self.folder_testing = folder_testing

    def on_epoch_end(self, epoch, logs={}):
        filepath = self.filepath.format(epoch=epoch, **logs)
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can save best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                              ' saving model to %s'
                              % (epoch, self.monitor, self.best,
                                 current, filepath))
                    self.best = current
                    self.model.save_weights(filepath, overwrite=True)

                    # Save the prediction results for testing set
                    if self.folder_testing:
                        filepath_testing = "{}/{:05d}.csv".format(self.folder_testing, epoch)

                        proba = self.model.predict_proba(self.testing_x)

                        pd.DataFrame({"ID": self.testing_id, "PredictedProb": [prob[0] if prob[0] else 0.0 for prob in proba]}).to_csv(filepath_testing, index=False)
                        print "Saving prediction results in {}".format(filepath_testing)
                else:
                    if self.verbose > 0:
                        print('Epoch %05d: %s did not improve' %
                              (epoch, self.monitor))
        else:
            if self.verbose > 0:
                print('Epoch %05d: saving model to %s' % (epoch, filepath))
            self.model.save_weights(filepath, overwrite=True)

def get_newest_model(folder):
    import glob

    newest = None
    try:
        newest = max(glob.iglob("{}/*.weights.hdf5".format(folder)), key=lambda x: os.path.getmtime(x))
    except:
        pass

    return newest

def lr(model_folder, layer, mini_batch, dimension,
       train_x, train_y, class_weight={0: 1, 1: 1}, origin_train=None,
       testing_data=None, testing_id=None,
       learning_rate=1e-6, dropout_rate=0.5, nepoch=10, activate_function="sigmoid"):
    number_of_feature = len(train_x[0])

    ori_train_x, ori_train_y = train_x, train_y
    if origin_train:
        ori_train_x, ori_train_y = origin_train

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

        print "Load weights from {}".format(filepath_model)

    checkpointer = KaggleCheckpoint(filepath=model_folder+"/{epoch}.weights.hdf5",
                                    testing_set=(testing_data, testing_id),
                                    folder_testing=model_folder,
                                    verbose=1, save_best_only=True)
    model.fit(train_x, train_y, nb_epoch=nepoch, batch_size=mini_batch, validation_split=0.1, class_weight=class_weight, callbacks=[checkpointer])

    return model

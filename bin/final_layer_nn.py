#!/usr/bin/env python

import os
import sys

import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

sys.path.append("../lib")
from deep_learning import KaggleCheckpoint

df = pd.read_csv(sys.argv[1])

cols = df.columns
cols = [col for col in cols if col.find("layer=2") > -1]

train_x = df[cols].values
train_y = df["target"].values

print train_x[0:10]
print train_y[0:10]

df = pd.read_csv(sys.argv[2])
test_id = df["ID"].values
test_x = df[cols].values

nepoch = 2000
batch_size = 4096
n_dim = len(train_x[0])
neurno = 64

folder = "./nn_dim={}_neurno={}".format(n_dim, neurno)
checkpointer = KaggleCheckpoint(filepath="{epoch}.weights.hdf5",
                                training_set=([train_x], train_x),
                                testing_set=([test_x], test_id),
                                folder=folder,
                                cost_string="auc",
                                verbose=0, save_best_only=True, save_training_dataset=False)

if not os.path.isdir(folder):
    os.makedirs(folder)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(neurno, input_dim=n_dim, init='uniform'))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(neurno, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1, init='uniform'))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(train_x, train_y, validation_split=0.1, nb_epoch=nepoch, batch_size=batch_size, class_weight={0: 1.5, 1: 1}, callbacks=[checkpointer])

prediction = [prob[0] if prob else 0.0 for prob in model.predict_proba(test_x)]

pd.DataFrame({"ID": test_id, "target": prediction}).to_csv("only_regressor_nn_layer=3_dim={}_neurno={}.submission.csv".format(n_dim, neurno), index=False)

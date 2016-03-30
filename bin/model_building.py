#!/usr/bin/env python

import os
import sys

import click
import numpy as np

sys.path.append("{}/../lib".format(os.path.dirname(os.path.abspath(__file__))))
from utils import log, INFO
from load import data_load, data_transform_1, data_transform_2, save_kaggle_submission, save_cache, load_cache
from learning import LearningFactory
from deep_learning import KaggleCheckpoint
from keras.callbacks import EarlyStopping
from ensemble_learning import layer_one_model, layer_two_model

BASEPATH = os.path.dirname(os.path.abspath(__file__))

@click.command()
@click.option("--deep-dimension", default=100, help="Dimension of Hiddle Layer")
@click.option("--deep-layer", default=0, help="Number of Hidden Layer")
@click.option("--nfold", default=10, help="Number of fold")
@click.option("--estimators", default=100, help="Number of estimator")
@click.option("--thread", default=1, help="Number of thread")
def learning(thread, nfold, estimators, deep_dimension, deep_layer):
    drop_fields = []
    #drop_fields = ['v8','v23','v25','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128']

    N = 650 - len(drop_fields)

    filepath_training = "{}/../input/train.csv".format(BASEPATH)
    filepath_testing = "{}/../input/test.csv".format(BASEPATH)

    filepath_cache_1 = "../input/{}_training_dataset.cache".format(N)

    train_x, test_x, train_y, test_id = None, None, None, None
    if os.path.exists(filepath_cache_1):
        train_x, test_x, train_y, test_id = load_cache(filepath_cache_1)
    else:
        train_x, test_x, train_y, test_id = data_transform_2(filepath_training, filepath_testing, drop_fields)
        save_cache((train_x, test_x, train_y, test_id), filepath_cache_1)

    train_X, test_X = train_x.values, test_x.values
    train_y = train_y.values
    test_id = test_id.values
    train_Y = train_y.astype(float)
    number_of_feature = len(train_X[0])

    # Init the parameters
    LearningFactory.set_n_estimators(estimators)

    models = ["shallow_gridsearch_extratree_regressor",
              "shallow_gridsearch_extratree_classifier",
              "shallow_gridsearch_randomforest_regressor",
              "shallow_gridsearch_randomforest_classifier",
              "shallow_xgboosting_regressor", #The logloss value is always nan, why???
              "shallow_xgboosting_classifier",
              #"shallow_gradientboosting_regressor",
              #"shallow_gradientboosting_classifier"
              ]
    layer2_model_name = "shallow_gridsearch_logistic_regressor"
    if deep_layer > 1:
        layer2_model_name = "deep_logistic_regressor"

    model_folder = "{}/../prediction_model/ensemble_learning/nfold={}_models={}_feature={}_estimators={}".format(\
                        BASEPATH, nfold, len(models), number_of_feature, estimators)

    print "Data Distribution is ({}, {}), and then the number of feature is {}".format(np.sum(train_Y==0), np.sum(train_Y==1), number_of_feature),

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    print "prepare to save data in {}".format(model_folder)

    # Phase 1. --> Model Training
    layer_2_train_x, layer_2_test_x, learning_loss = layer_one_model(model_folder, train_X, train_Y, test_X, test_id, models, layer2_model_name, n_folds=nfold, number_of_thread=thread,
                             filepath_queue="{}/queue.pickle".format(model_folder), filepath_nfold="{}/nfold.pickle".format(model_folder))

    # Phase 2. --> Model Training
    deep_setting = {}
    if deep_layer > 1:
        deep_learning_model_folder = "{}/nn_layer={}_dimension={}".format(model_folder, deep_layer, deep_dimension)
        if not os.path.isdir(deep_learning_model_folder):
            os.makedirs(deep_learning_model_folder)

        checkpointer = KaggleCheckpoint(filepath=deep_learning_model_folder + "/{epoch}.weights.hdf5",
                                        training_set=([layer_2_train_x, layer_2_train_x], train_Y),
                                        testing_set=([layer_2_test_x, layer_2_test_x], test_id),
                                        folder=deep_learning_model_folder,
                                        verbose=1, save_best_only=True)

        early_stopping = EarlyStopping(monitor='binary_crossentropy', patience=20, verbose=0, mode='auto')

        deep_setting["folder_weights"] = deep_learning_model_folder
        deep_setting["batch_size"] = 64
        deep_setting["number_of_layer"] = 3
        deep_setting["dimension"] = deep_layer
        deep_setting["callbacks"] = [checkpointer]
        deep_setting["nepoch"] = 10000
        deep_setting["validation_split"] = 0.15
        deep_setting["class_weight"] = {0: 2, 1: 1}

        model_folder = deep_learning_model_folder

    results = layer_two_model(models, layer_2_train_x, train_Y, test_id, layer_2_test_x, learning_loss, layer2_model_name,
                              "{}/training.csv".format(model_folder), "{}/testing.csv".format(model_folder), "{}/logloss.pickle".format(model_folder),
                              deep_setting)

    # Save the submission CSV file
    filepath_output = "{}/kaggle_BNP_submission_{}.csv".format(model_folder, layer2_model_name)
    save_kaggle_submission(test_id, results, filepath_output)

if __name__ == "__main__":
    learning()

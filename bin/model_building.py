#!/usr/bin/env python

#############################################################################
#
# version 2.0: add feature from KMeans cluster 2016/03/31
#
#############################################################################

import os
import sys

import click
import numpy as np

sys.path.append("{}/../lib".format(os.path.dirname(os.path.abspath(__file__))))
from utils import log, INFO
from load import load_data, load_advanced_data, data_polynomial, save_kaggle_submission
from learning import LearningFactory
from deep_learning import KaggleCheckpoint
from keras.callbacks import EarlyStopping
from ensemble_learning import layer_one_model, layer_two_model, get_max_mean_min_probabilities

BASEPATH = os.path.dirname(os.path.abspath(__file__))

@click.command()
@click.option("--nfold", default=10, help="Number of fold")
@click.option("--estimators", default=100, help="Number of estimator")
@click.option("--thread", default=1, help="Number of thread")
@click.option("--weight", default=1, help="Weight of Class 0")
@click.option("--polynomial", is_flag=True, help="Use Polynomial")
@click.option("--kmeans", is_flag=True, help="Use the features of Kmeans")
def learning(thread, nfold, estimators, weight, polynomial, kmeans):
    drop_fields = []
    #drop_fields = ['v8','v23','v25','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128']

    N = 650 - len(drop_fields)

    filepath_training = "{}/../input/train.csv".format(BASEPATH)
    filepath_testing = "{}/../input/test.csv".format(BASEPATH)
    filepath_cache_1 = "../input/{}_training_dataset.cache".format(N)

    train_x, test_x, train_y, test_id, train_id = load_data(filepath_cache_1, filepath_training, filepath_testing, drop_fields)

    train_X, test_X = train_x.values, test_x.values
    if polynomial:
        filepath_cache = "{}/../input/polynomial.pickle".format(BASEPATH)
        train_X, test_X = data_polynomial(filepath_cache, train_X, test_X)

    train_y = train_y.values
    test_id = test_id.values
    train_Y = train_y.astype(float)
    number_of_feature = len(train_X[0])

    # Init the parameters
    LearningFactory.set_n_estimators(estimators)

    # Init the parameters of cluster
    cluster_kmeans4_setting = {"n_clusters": 4, "n_init": 10, "random_state": 1201}

    # Init the parameters of deep learning
    deep_learning_model_folder = "{}/../prediction_model/deep_learning/layer=3_neurno=2000".format(BASEPATH)
    if not os.path.isdir(deep_learning_model_folder):
        os.makedirs(deep_learning_model_folder)

    training_dataset, testing_dataset = [train_X], [test_X]
    checkpointer = KaggleCheckpoint(filepath=deep_learning_model_folder  + "/{epoch}.weights.hdf5",
                                    training_set=(training_dataset, train_Y),
                                    testing_set=(testing_dataset, test_id),
                                    folder=None,
                                    verbose=1, save_best_only=True)

    deep_layer3_neurno2000_setting = {"folder": deep_learning_model_folder,
                                      "input_dims": number_of_feature,
                                      "batch_size": 128,
                                      "number_of_layer": 3,
                                      "dimension": 1000,
                                      "callbacks": [checkpointer],
                                      "nepoch": 10,
                                      "validation_split": 0,
                                      "class_weight": {0: weight, 1: 1}}

    models = [\
              ("shallow_gridsearch_extratree_regressor", None),
              #"shallow_gridsearch_extratree_classifier",
              #"shallow_gridsearch_randomforest_regressor",
              #"shallow_gridsearch_randomforest_classifier",
              #"shallow_xgboosting_regressor", #The logloss value is always nan, why???
              #"shallow_xgboosting_classifier",
              ("cluster_kmeans_16", cluster_kmeans4_setting),
              #"cluster_kmeans_64",
              #"cluster_kmeans_128",
              #"cluster_kmeans_512",
              ("deep_layer3_neuron2000", deep_layer3_neurno2000_setting),
              #"deep_layer5_neuron2000"
              #"shallow_gradientboosting_regressor",
              #"shallow_gradientboosting_classifier"
              ]

    layer2_model_name = "shallow_gridsearch_logistic_regressor"
    model_folder = "{}/../prediction_model/ensemble_learning/nfold={}_models={}_feature={}_estimators={}_polynomial={}".format(\
                        BASEPATH, nfold, len(models), number_of_feature, estimators, polynomial)

    print "Data Distribution is ({}, {}), and then the number of feature is {}".format(np.sum(train_Y==0), np.sum(train_Y==1), number_of_feature),

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    print "prepare to save data in {}".format(model_folder)

    # Phase 1. --> Model Training
    layer_2_train_x, layer_2_test_x, learning_loss = layer_one_model(model_folder, train_X, train_Y, test_X, test_id, models, layer2_model_name, n_folds=nfold, number_of_thread=thread,
                             filepath_queue="{}/queue.pickle".format(model_folder), filepath_nfold="{}/nfold.pickle".format(model_folder))

    # Phase 2. --> Model Training
    results = layer_two_model(models, layer_2_train_x, train_Y, test_id, layer_2_test_x, learning_loss, (layer2_model_name, None),
                              "{}/training.csv".format(model_folder), "{}/testing.csv".format(model_folder), "{}/logloss.pickle".format(model_folder))

    # Save the submission CSV file
    filepath_output = "{}/kaggle_BNP_submission_{}.csv".format(model_folder, layer2_model_name)
    save_kaggle_submission(test_id, results, filepath_output)

if __name__ == "__main__":
    learning()

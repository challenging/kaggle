#!/usr/bin/env python

import os
import sys

import click
import numpy as np

sys.path.append("{}/../lib".format(os.path.dirname(os.path.abspath(__file__))))
from utils import log, INFO
from load import data_load, data_transform_1, data_transform_2, save_kaggle_submission, pca, save_cache, load_cache
from learning import LearningFactory
from deep_learning import KaggleCheckpoint
from ensemble_learning import layer_one_model, layer_two_model

BASEPATH = os.path.dirname(os.path.abspath(__file__))

@click.command()
@click.option("--deep-dimension", default=100, help="Dimension of Hiddle Layer")
@click.option("--deep-layer", default=2, help="Number of Hidden Layer")
@click.option("--deep-layer2", is_flag=True, help="Use Deep in Layer-2")
@click.option("--nfold", default=10, help="Number of fold")
@click.option("--estimators", default=100, help="Number of estimator")
@click.option("--thread", default=1, help="Number of thread")
@click.option("--pca-number", default=None, help="Dimension of PCA")
@click.option("--transform", default=1, help="Tranform Methodology")
def learning(pca_number, transform, thread, nfold, estimators, deep_layer2, deep_dimension, deep_layer):
    filepath_cache_1 = "../input/455_dataset.cache"

    drop_fields = ['v8','v23','v25','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128']
    train_x, train_y, test_x, test_id = data_load(drop_fields=drop_fields, filepath_cache=filepath_cache_1)

    if transform == 1:
        train_x, test_x = data_transform_1(train_x, test_x)
    elif transform == 2:
        filepath = "{}/../input/transform2.csv".format(BASEPATH)
        train_x, test_x = data_transform_2(train_x, test_x, filepath)
    else:
        click.echo("Not Found the tranform metholody {}".format(transform))
        sys.exit(1)

    # Init the parameters
    LearningFactory.set_n_estimators(estimators)

    # PCA Data Transformation
    train_X = train_x.values
    train_pca_X, test_pca_X = None, None

    filepath_cache_2 = "../input/455_pcanumber={}_dataset.cache".format(pca_number)
    if os.path.exists(filepath_cache_2):
        train_pca_X, test_pca_X = load_cache(filepath_cache_2)

        log("Load data from cache file({}) for PCA".format(filepath_cache_2), INFO)
    else:
        if pca_number == 0:
            train_pca_X = train_X
            test_pca_X = test_x
        else:
            pca_model = pca(train_X, int(pca_number) if pca_number else None)
            train_pca_X = pca_model.fit_transform(train_X)
            test_pca_X = pca_model.fit_transform(test_x)

        save_cache((train_pca_X, test_pca_X), filepath_cache_2)
        log("Save data to cache file({}) for PCA".format(filepath_cache_2), INFO)

    train_Y = train_y.astype(float)

    number_of_feature = len(train_pca_X[0])
    if not pca_number:
        pca_number = number_of_feature

    models = ["shallow_gridsearch_extratree_regressor", "shallow_gridsearch_extratree_classifier",
              "shallow_gridsearch_randomforest_regressor", "shallow_gridsearch_randomforest_classifier",
              "shallow_xgboosting_regressor", #The logloss value is always nan, why???
              "shallow_xgboosting_classifier"]
              #"shallow_gridsearch_gradientboosting_regressor", "shallow_gridsearch_gradientboosting_classifier"]
    layer2_model_name = "shallow_gridsearch_logistic_regressor"

    model_folder = "{}/../prediction_model/ensemble_learning/transform={}_models={}_feature={}_pcanumber={}_estimators={}".format(\
                        BASEPATH, transform, len(models), number_of_feature, pca_number, estimators)

    print "Data Distribution is ({}, {}), and then the number of feature is {}".format(np.sum(train_Y==0), np.sum(train_Y==1), number_of_feature),

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    print "prepare to save data in {}".format(model_folder)

    # Phase 1. --> Model Training
    layer_2_train_x, layer_2_test_x, learning_loss = layer_one_model(model_folder, train_pca_X, train_Y, test_pca_X, test_id, models, layer2_model_name, n_folds=nfold, number_of_thread=thread,
                             filepath_queue="{}/queue.pickle".format(model_folder), filepath_nfold="{}/nfold.pickle".format(model_folder))

    # Phase 2. --> Model Training
    deep_setting = {}
    if deep_layer2:
        checkpointer = KaggleCheckpoint(filepath=model_folder+"/{epoch}.weights.hdf5",
                                        testing_set=(test_x, test_id),
                                        folder_testing=model_folder,
                                        verbose=1, save_best_only=True)

        deep_setting["mini_batch"] = 5
        deep_setting["number_of_layer"] = 3
        deep_setting["dimension"] = 1000
        deep_setting["callbacks"] = [checkpointer]
        deep_setting["nepoch"] = 1000

    results = layer_two_model(models, layer_2_train_x, train_Y, layer_2_test_x, learning_loss, layer2_model_name, model_folder, deep_setting)

    # Save the submission CSV file
    filepath_output = "{}/kaggle_BNP_submission_{}.csv".format(model_folder, layer2_model_name)
    save_kaggle_submission(test_id, results, filepath_output)

if __name__ == "__main__":
    learning()

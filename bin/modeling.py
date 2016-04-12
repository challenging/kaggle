#!/usr/bin/env python

#############################################################################
#
# version 2.0: add feature from KMeans cluster 2016/03/31
#
#############################################################################

import os
import sys
import click
import ConfigParser

import numpy as np

sys.path.append("{}/../lib".format(os.path.dirname(os.path.abspath(__file__))))
from utils import log, INFO
from load import load_data, load_advanced_data, load_cache, save_cache, save_kaggle_submission, load_interaction_information
from learning import LearningFactory
from configuration import ModelConfParser
from deep_learning import KaggleCheckpoint
from keras.callbacks import EarlyStopping
from ensemble_learning import layer_one_model, layer_two_model, get_max_mean_min_probabilities

BASEPATH = os.path.dirname(os.path.abspath(__file__))

@click.command()
@click.option("--conf", is_require=True, help="Configureation File to this model")
@click.option("--thread", default=1, help="Number of thread")
def learning(conf, thread):
    drop_fields = []
    N = 650 - len(drop_fields)

    parser = ModelConfParser(conf)

    BASEPATH = parser.get_workspace()
    binsize, topX = parser.get_interaction_information()

    filepath_training = "{}/input/train.csv".format(BASEPATH)
    filepath_testing = "{}/input/test.csv".format(BASEPATH)
    filepath_cache_1 = "{}/input/{}_training_dataset.cache".format(BASEPATH, N)
    filepath_ii = "{}/input/transform2=True_testing=-1_type=2_binsize={}_combination=2.pkl".format(BASEPATH, binsize)
    filepath_cache_ii = "{}/input/transform2=True_testing=-1_type=2_binsize={}_combination=2.cache.pkl".format(BASEPATH, binsize)

    train_x, test_x, train_y, test_id, train_id = load_data(filepath_cache_1, filepath_training, filepath_testing, drop_fields)

    if interaction_information:
        if os.path.exists(filepath_cache_ii):
            train_x, test_x = load_cache(filepath_cache_ii)
        else:
            for (layer1, layer2), value in load_interaction_information(filepath_ii, topX):
                train_x["{}-{}".format(layer1, layer2)] = train_x[layer1].values * train_x[layer2].values * value
                test_x["{}-{}".format(layer1, layer2)] = test_x[layer1].values * test_x[layer2].values * value

            save_cache((train_x, test_x), filepath_cache_ii)

    train_X, test_X = train_x.values, test_x.values

    train_y = train_y.values
    test_id = test_id.values
    train_Y = train_y.astype(float)
    number_of_feature = len(train_X[0])

    # Init the parameters
    models = []

    # Init the parameters of deep learning
    training_dataset, testing_dataset = [train_X], [test_X]
    checkpointer = KaggleCheckpoint(filepath="{epoch}.weights.hdf5",
                                    training_set=(training_dataset, train_Y),
                                    testing_set=(testing_dataset, test_id),
                                    folder=None,
                                    cost_string=cost,
                                    verbose=0, save_best_only=False)

    # Init the parameters of cluster
    for model_section in parser.get_layer_models():
        setting = parser.get_model_setting(model_setting)

        method = setting["method"]
        del setting["method"]

        if "class_weight" in setting:
            setting["class_weight"] = {0: setting["class_weight"], 1: 1}

        if method.find("deep") > -1:
            setting["folder"] = None
            setting["input_dims"] = number_of_feature
            setting["callbacks"] = [checkpointer]
            setting["number_of_layer"] = setting.pop("layer_number")
            setting["dimension"] = setting.pop("layer_dimension")

        models.append((method, setting))

    layer2_model_name = "shallow_gridsearch_logistic_regressor"
    model_folder = "{}/prediction_model/ensemble_learning/nfold={}_models={}_feature={}_estimators={}_binsize={}_topX={}".format(\
                        BASEPATH, nfold, len(models), number_of_feature, estimators, binsize, topX)

    print "Data Distribution is ({}, {}), and then the number of feature is {}, and then prepare to save data in {}".format(np.sum(train_Y==0), np.sum(train_Y==1), number_of_feature, model_folder),

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # Phase 1. --> Model Training
    layer_2_train_x, layer_2_test_x, learning_loss = layer_one_model(model_folder, train_X, train_Y, test_X, test_id, models, layer2_model_name, cost_string=cost, n_folds=nfold, number_of_thread=thread,
                             filepath_queue="{}/queue.pickle".format(model_folder), filepath_nfold="{}/nfold.pickle".format(model_folder))

    # Phase 2. --> Model Training
    results = layer_two_model(models, layer_2_train_x, train_Y, test_id, layer_2_test_x, learning_loss, (layer2_model_name, None),
                              "{}/training.csv".format(model_folder), "{}/testing.csv".format(model_folder), "{}/logloss.pickle".format(model_folder), cost_string=cost)

    # Save the submission CSV file
    filepath_output = "{}/kaggle_BNP_submission_{}.csv".format(model_folder, layer2_model_name)
    save_kaggle_submission(test_id, results, filepath_output)

if __name__ == "__main__":
    learning()

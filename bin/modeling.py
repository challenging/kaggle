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
from ensemble_learning import layer_one_model, layer_two_model, layer_three_model

BASEPATH = os.path.dirname(os.path.abspath(__file__))

@click.command()
@click.option("--conf", required=True, help="Configureation File to this model")
@click.option("--thread", default=1, help="Number of thread")
def learning(conf, thread):
    drop_fields = []
    N = 650 - len(drop_fields)

    parser = ModelConfParser(conf)

    BASEPATH = parser.get_workspace()
    binsize, topX = parser.get_interaction_information()
    cost = parser.get_cost()
    nfold = parser.get_nfold()

    filepath_training = "{}/input/train.csv".format(BASEPATH)
    filepath_testing = "{}/input/test.csv".format(BASEPATH)
    filepath_cache_1 = "{}/input/{}_training_dataset.cache".format(BASEPATH, N)
    filepath_ii = "{}/input/transform2=True_testing=-1_type=2_binsize={}_combination=2.pkl".format(BASEPATH, binsize)
    filepath_cache_ii = "{}/input/transform2=True_testing=-1_type=2_binsize={}_combination=2.cache.pkl".format(BASEPATH, binsize)

    train_x, test_x, train_y, test_id, train_id = load_data(filepath_cache_1, filepath_training, filepath_testing, drop_fields)

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

    layer1_models, layer2_models = [], []

    # Init the parameters of deep learning
    training_dataset, testing_dataset = [train_X], [test_X]
    checkpointer = KaggleCheckpoint(filepath="{epoch}.weights.hdf5",
                                    training_set=(training_dataset, train_Y),
                                    testing_set=(testing_dataset, test_id),
                                    folder=None,
                                    cost_string=cost,
                                    verbose=0, save_best_only=False)

    # Init the parameters of cluster
    for model_section in parser.get_layer_models(1):
        method, setting = parser.get_model_setting(model_section)

        if "class_weight" in setting:
            if isinstance(setting["class_weight"], int) or isinstance(setting["class_weight"]. float):
                setting["class_weight"] = {0: setting["class_weight"], 1: 1}
            else:
                setting["class_weight"] = "balanced"

        if method.find("deep") > -1:
            setting["folder"] = None
            setting["input_dims"] = number_of_feature
            setting["callbacks"] = [checkpointer]
            setting["number_of_layer"] = setting.pop("layer_number")
            setting["dimension"] = setting.pop("layer_dimension")

            del setting["n_jobs"]

        log(setting, INFO)

        layer1_models.append((method, setting))
        log("Get the configuration of {} from {}".format(method, conf), INFO)

    for model_section in parser.get_layer_models(2):
        method, setting = parser.get_model_setting(model_section)

        layer2_models.append((method, setting))

    model_folder = "{}/prediction_model/ensemble_learning/nfold={}_layer1={}_layer2={}_feature={}_binsize={}_topX={}".format(\
                        BASEPATH, nfold, len(layer1_models), len(layer2_models), number_of_feature, binsize, topX)

    print "Data Distribution is ({}, {}), and then the number of feature is {}, and then prepare to save data in {}".format(np.sum(train_Y==0), np.sum(train_Y==1), number_of_feature, model_folder),

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # Phase 1. --> Model Training
    layer_2_train_x, layer_2_test_x, learning_loss = layer_one_model(model_folder, train_X, train_Y, test_X, layer1_models, cost_string=cost, n_folds=nfold, number_of_thread=thread,
                             filepath_queue="{}/queue.pickle".format(model_folder), filepath_nfold="{}/nfold.pickle".format(model_folder))

    # Phase 2. --> Model Training
    layer3_train_x, layer3_test_x = layer_two_model(layer1_models, layer_2_train_x, train_Y, layer_2_test_x, learning_loss, layer2_models,
                              "{}/training.csv".format(model_folder), cost_string=cost)

    # Phase 3. --> Model Training
    submission_results = layer_three_model(layer3_train_x, train_Y, layer3_test_x, cost_string=cost)

    '''
    # Save the cost
    save_cache(learning_loss, "{}/logloss.pickle".format(model_folder))

    # Save the submission CSV file
    filepath_output = "{}/kaggle_BNP_submission_{}.csv".format(model_folder)
    save_kaggle_submission(test_id, results, filepath_output)
    '''

if __name__ == "__main__":
    learning()

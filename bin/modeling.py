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
from ensemble_learning import layer_model, final_model, store_layer_output

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
    folder_ii = "{}/input/interaction_information/transform2=True_testing=-1_binsize={}".format(BASEPATH, binsize)

    train_x, test_x, train_y, test_id, train_id = load_data(filepath_cache_1, filepath_training, filepath_testing, drop_fields)

    columns = train_x.columns
    for layers, value in load_interaction_information(folder_ii, top):
        for df in [train_x, test_x]:
            t = value
            breaking_layer = None
            for layer in layers:
                if layer in columns:
                    t *= df[layer]
                else:
                    breaking_layer = layer
                    break

            if breaking_layer != None:
                df[";".join(layers)] = t
            else:
                log("Skip {} due to {} not in columns".format(layers, breaking_layer), WARN)
                break

    train_X, test_X = train_x.values, test_x.values

    train_y = train_y.values
    test_id = test_id.values
    train_Y = train_y.astype(float)
    number_of_feature = len(train_X[0])

    layer1_models, layer2_models, last_model = [], [], []

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
            if isinstance(setting["class_weight"], int) or isinstance(setting["class_weight"], float):
                setting["class_weight"] = {0: setting["class_weight"], 1: 1}
            else:
                setting["class_weight"] = "balanced"

        if method.find("deep") > -1:
            setting["folder"] = None
            setting["input_dims"] = number_of_feature
            setting["callbacks"] = [checkpointer]
            setting["number_of_layer"] = setting.pop("layer_number")
            setting["dimension"] = int(number_of_feature*1.25)

            del setting["n_jobs"]

        layer1_models.append((method, setting))
        log("Get the configuration of {} from {}".format(method, conf), INFO)

    for model_section in parser.get_layer_models(2):
        method, setting = parser.get_model_setting(model_section)

        setting["cost"] = parser.get_cost()

        log(setting)

        layer2_models.append((method, setting))

    for model_section in parser.get_layer_models(3):
        method, setting = parser.get_model_setting(model_section)

        last_model.append((method, setting))

    model_folder = "{}/prediction_model/ensemble_learning/nfold={}_layer1={}_layer2={}_feature={}_binsize={}_topX={}".format(\
                        BASEPATH, nfold, len(layer1_models), len(layer2_models), number_of_feature, binsize, topX)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    filepath_training = "{}/training_proba_tracking.csv".format(model_folder)
    filepath_testing = "{}/testing_proba_tracking.csv".format(model_folder)

    # Phase 1. --> Model Training
    filepath_queue = "{}/layer1_queue.pkl".format(model_folder)
    filepath_nfold = "{}/layer1_nfold.pkl".format(model_folder)
    layer2_train_x, layer2_test_x, learning_loss = layer_model(model_folder, train_X, train_Y, test_X, layer1_models,
                             filepath_queue, filepath_nfold,
                             n_folds=nfold, number_of_thread=thread)

    # Phase 2. --> Model Training
    filepath_queue = "{}/layer2_queue.pkl".format(model_folder)
    filepath_nfold = "{}/layer2_nfold.pkl".format(model_folder)
    layer3_train_x, layer3_test_x, learning_loss = layer_model(model_folder, layer2_train_x, train_Y, layer2_test_x, layer2_models,
                             filepath_queue, filepath_nfold,
                             n_folds=nfold, number_of_thread=thread)

    training_dataset_proba = np.hstack((layer2_train_x, layer3_train_x))
    training_targets = [{"Target": train_y}]
    store_layer_output([m[0] for m in layer1_models+layer2_models], training_dataset_proba, filepath_training, optional=training_targets)

    testing_dataset_proba = np.hstack((layer2_test_x, layer3_test_x))
    testing_targets = [{"ID": test_id}]
    store_layer_output([m[0] for m in layer1_models+layer2_models], testing_dataset_proba, filepath_testing, optional=testing_targets)

    # Phase 3. --> Model Training
    submission_results = final_model(last_model[0], layer3_train_x, train_Y, layer3_test_x, cost_string=cost)

    '''
    # Save the cost
    save_cache(learning_loss, "{}/logloss.pickle".format(model_folder))

    # Save the submission CSV file
    filepath_output = "{}/kaggle_BNP_submission_{}.csv".format(model_folder)
    save_kaggle_submission(test_id, results, filepath_output)
    '''

if __name__ == "__main__":
    learning()

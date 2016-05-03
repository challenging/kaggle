#!/usr/bin/env python

#############################################################################
#
# version 2.0: add feature from KMeans cluster 2016/03/31
#
#############################################################################

import os
import sys
import click
import shutil

import numpy as np

sys.path.append("{}/../lib".format(os.path.dirname(os.path.abspath(__file__))))
from utils import make_a_stamp, log, INFO, WARN
from load import load_data, load_advanced_data, load_cache, save_cache, save_kaggle_submission, load_interaction_information, load_feature_importance
from learning import LearningFactory
from configuration import ModelConfParser
from deep_learning import KaggleCheckpoint
from keras.callbacks import EarlyStopping
from ensemble_learning import layer_model, final_model, store_layer_output

BASEPATH = os.path.dirname(os.path.abspath(__file__))

@click.command()
@click.option("--conf", required=True, help="Configureation File to this model")
@click.option("--thread", default=1, help="Number of thread")
@click.option("--is-testing", is_flag=True, help="Turn on the testing mode")
def learning(conf, thread, is_testing):
    drop_fields = []

    parser = ModelConfParser(conf)

    BASEPATH = parser.get_workspace()
    objective = parser.get_objective()
    binsize, top = parser.get_interaction_information()
    cost = parser.get_cost()
    nfold = parser.get_nfold()
    top_feature = parser.get_top_feature()

    filepath_training = "{}/input/train.csv".format(BASEPATH)
    filepath_testing = "{}/input/test.csv".format(BASEPATH)
    filepath_cache_1 = "{}/input/train.pkl".format(BASEPATH)
    folder_ii = "{}/input/interaction_information/transform2=True_testing=-1_binsize={}".format(BASEPATH, binsize)
    filepath_feature_importance = "{}/etc/feature_profile/transform2=True_binsize={}_top={}.pkl".format(BASEPATH, binsize, top)

    train_x, test_x, train_y, test_id, train_id = load_data(filepath_cache_1, filepath_training, filepath_testing, drop_fields)
    if is_testing:
        train_x = train_x.head(1000)
        train_y = train_y.head(1000)
    basic_columns = train_x.columns

    for layers, value in load_interaction_information(folder_ii, threshold=str(top_feature)):
        for df in [train_x, test_x]:
            t = value
            breaking_layer = None
            for layer in layers:
                if layer in basic_columns:
                    t *= df[layer].values
                else:
                    breaking_layer = layer
                    break

            if breaking_layer == None:
                df[";".join(layers)] = t
            else:
                log("Skip {} due to {} not in columns".format(layers, breaking_layer), WARN)
                break

    ii_columns = train_x.columns
    importance_columns = load_feature_importance(filepath_feature_importance, top_feature)

    predictors = {"basic": basic_columns,
                  "interaction-information-3": [column for column in ii_columns if column.count(";") == 1],
                  "interaction-information-4": [column for column in ii_columns if column.count(";") == 2],
                  "feature-importance": importance_columns}

    train_y = train_y.values
    test_id = test_id.values
    train_Y = train_y.astype(float)

    layer1_models, layer2_models, last_model = [], [], []
    data_dimension = []

    # Init the parameters of deep learning
    checkpointer = KaggleCheckpoint(filepath="{epoch}.weights.hdf5",
                                    training_set=([train_x], train_Y),
                                    testing_set=([test_x], test_id),
                                    folder=None,
                                    cost_string=cost,
                                    verbose=0, save_best_only=True, save_training_dataset=False)

    # Init the parameters of cluster
    for idx, layer_models in enumerate([layer1_models, layer2_models, last_model]):
        data_dimension.append([])

        for model_section in parser.get_layer_models(idx+1):
            for method, setting in parser.get_model_setting(model_section):
                if method.find("deep") > -1:
                    setting["folder"] = None

                    if "data_dimension" in setting:
                        if setting["data_dimension"] == "basic":
                            setting["input_dims"] = len(basic_columns)
                        elif setting["data_dimension"] == "importance":
                            setting["input_dims"] = len(importance_columns)
                        elif setting["data_dimension"].find("interaction-information") != -1:
                            setting["input_dims"] = top_feature
                        else:
                            log("Wrong Setting for input_dims because the data_dimension is {}".format(setting["data_dimension"]), ERRPR)
                            sys.exit(100)

                        data_dimension[idx].append(setting["data_dimension"])
                    else:
                        log("Not found data_dimension in LAYER{}".format(idx+1), INFO)

                        data_dimension[idx].append("all")

                    setting["callbacks"] = [checkpointer]
                    setting["number_of_layer"] = setting.pop("layer_number")
                else:
                    if "data_dimension" in setting:
                        data_dimension[idx].append(setting["data_dimension"])
                    else:
                        data_dimension[idx].append("all")

                layer_models.append((method, setting))
                log("Get the configuration of {} from {}".format(method, conf), INFO)
                log("The setting is {}".format(setting), INFO)

    folder_model = "{}/prediction_model/ensemble_learning/conf={}_is_testing={}_nfold={}_layer1={}_layer2={}_binsize={}_top={}".format(\
                        BASEPATH, os.path.basename(conf), is_testing, nfold, len(layer1_models), len(layer2_models), binsize, top_feature)
    folder_middle = "{}/etc/middle_layer/is_testing={}_nfold={}_binsize={}_top={}".format(\
                        BASEPATH, is_testing, nfold, binsize, top_feature)

    if is_testing and os.path.isdir(folder_model):
        log("Due to the testing mode, remove the {} firstly".format(folder_model), INFO)
        shutil.rmtree(folder_model)

    folder_submission = "{}/submission".format(folder_model)
    if not os.path.isdir(folder_submission):
        os.makedirs(folder_submission)

    filepath_training = "{}/training_proba_tracking.csv".format(folder_model)
    filepath_testing = "{}/testing_proba_tracking.csv".format(folder_model)

    previous_training_dataset, previous_testing_dataset = train_x, test_x
    prediction_testing_history, prediction_training_history, learning_loss_history = {"ID": test_id}, {"Target": train_Y}, []

    # Model Training
    m = [layer1_models, layer2_models, last_model]
    for idx, models in enumerate(m):
        filepath_queue = "{}/layer{}_queue.pkl".format(idx+1, folder_model)
        filepath_nfold = "{}/layer{}_nfold.pkl".format(idx+1, folder_model)
        layer_train_x, layer_test_x, learning_loss = layer_model(\
                                 objective, folder_model, folder_middle, predictors, previous_training_dataset, train_Y, previous_testing_dataset, models,
                                 filepath_queue, filepath_nfold,
                                 n_folds=(1 if idx==len(m)-1 else nfold), cost_string=cost, number_of_thread=thread, saving_results=(True if idx==0 else False))

        learning_loss_history.append(learning_loss)

        col = layer_test_x.shape[1]
        for idx_col in range(0, col):
            submission = layer_test_x[:,idx_col]
            filepath_submission = "{}/layer={}_dimension={}_model={}_params={}.csv".format(folder_submission, idx+1, data_dimension[idx][idx_col], models[idx_col][0], make_a_stamp(models[idx_col][1]))
            save_kaggle_submission({"ID": test_id, "Target": submission}, filepath_submission)

            prediction_training_history["layer={}_method={}".format(idx+1, models[idx_col][0])] = layer_train_x[:, idx_col]
            prediction_testing_history["layer={}_method={}".format(idx+1, models[idx_col][0])] = layer_test_x[:, idx_col]

        previous_training_dataset = layer_train_x
        previous_testing_dataset = layer_test_x

        log("Layer{} is done...".format(idx+1), INFO)

    filepath_history_training_prediction = "{}/history_training.csv".format(folder_model)
    save_kaggle_submission(prediction_training_history, filepath_history_training_prediction)

    filepath_history_testing_prediction = "{}/history_testing.csv".format(folder_model)
    save_kaggle_submission(prediction_testing_history, filepath_history_testing_prediction)

    filepath_history_learning_loss = "{}/learning_loss.pkl".format(folder_model)

if __name__ == "__main__":
    learning()

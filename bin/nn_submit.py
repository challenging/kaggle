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
from utils import log, INFO, WARN
from load import load_data, load_advanced_data, load_cache, save_cache, save_kaggle_submission, load_interaction_information, load_feature_importance
from learning import LearningFactory
from configuration import ModelConfParser

BASEPATH = os.path.dirname(os.path.abspath(__file__))

@click.command()
@click.option("--conf", required=True, help="Configureation File to this model")
@click.option("--thread", default=1, help="Number of thread")
@click.option("--is-feature-importance", is_flag=True, help="Turn on the feature importance")
@click.option("--is-testing", is_flag=True, help="Turn on the testing mode")
def nn_submit(conf, thread, is_feature_importance, is_testing):
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

    columns = train_x.columns
    for layers, value in load_interaction_information(folder_ii, threshold=top):
        for df in [train_x, test_x]:
            t = value
            breaking_layer = None
            for layer in layers:
                if layer in columns:
                    t *= df[layer].values
                else:
                    breaking_layer = layer
                    break

            if breaking_layer == None:
                df[";".join(layers)] = t
            else:
                log("Skip {} due to {} not in columns".format(layers, breaking_layer), WARN)
                break

    columns = train_x.columns
    if is_feature_importance:
        predictors = load_feature_importance(filepath_feature_importance, top_feature)

        drop_fields = [column for column in columns if column not in predictors]
        log("Due to the opening of feature importance so dropping {} columns".format(len(drop_fields)), INFO)

        train_x = train_x.drop(drop_fields, axis=1)
        test_x = test_x.drop(drop_fields, axis=1)

    train_X, test_X = train_x.values, test_x.values

    train_y = train_y.values
    test_id = test_id.values
    train_Y = train_y.astype(float)
    number_of_feature = len(train_X[0])

    # Init the parameters of cluster
    method, setting = parser.get_model_setting("LAYER1_MODEL1")
    setting["nfold"] = 0
    setting["input_dims"] = number_of_feature
    setting["number_of_layer"] = setting.pop("layer_number")

    model_folder = "{}/prediction_model/ensemble_learning/is_testing={}_is_feature_importance={}_nfold={}_layer1={}_layer2={}_feature={}_binsize={}_top={}/".format(\
                        BASEPATH, is_testing, is_feature_importance, nfold, 1, 2, number_of_feature, binsize, top)

    setting["folder"] = model_folder

    proba = LearningFactory.get_model(objective, (method, setting), cost).predict(test_X)

    results = {"ID": test_id, "Target": proba}
    save_kaggle_submission(results, "{}/nn_submission.csv".format(model_folder))

if __name__ == "__main__":
    nn_submit()

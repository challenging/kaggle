#!/usr/bin/env python

import os
import sys
import time

import numpy as np
import pandas as pd

from load import load_cache, save_cache
from utils import log, DEBUG, INFO, WARN
from cluster import ClusterFactory, Cluster

BASEPATH = os.path.dirname(os.path.abspath(__file__))

def layer1_tag_labels(model_folder, n_features, methodology, train_x, test_x, setting={}):
    timestamp_start = time.time()
    cluster_model = None
    filepath_model = "{}/{}_nclusters={}_nfeature={}.cache".format(model_folder, methodology, setting["n_clusters"], n_features)
    if os.path.exists(filepath_model):
        model = load_cache(filepath_model)

        cluster_model = Cluster(methodology, model)
    else:
        log("Not Found cache file, {}".format(filepath_model), INFO)

        cluster_model = ClusterFactory.get_model(methodology, setting)
        cluster_model.fit(train_x)

        save_cache(cluster_model.model, filepath_model)

    log("Cost {} secends to build {} model".format(time.time() - timestamp_start, methodology), INFO)

    training_labels = cluster_model.get_labels()
    testing_labels = cluster_model.predict(test_x)

    return training_labels, testing_labels

def layer2_calculation(labels, train_y):
    ratio = {}

    timestamp_start = time.time()
    for idx, target in enumerate(train_y):
        label = labels[idx]

        ratio.setdefault(label, [0, 0])
        ratio[label][int(target)] += 1

    for label, nums in ratio.items():
        target_0, target_1 = nums[0], nums[1]

        ratio[label] = float(target_1) / (target_0 + target_1)

    log("Cost {} secends to calculate the ratio".format(time.time() - timestamp_start), INFO)

    return ratio

def layer_process(model_folder, n_features, methodology, train_x, train_y, train_id, test_x, test_id, setting={}):
    training_labels, testing_labels = layer1_tag_labels(model_folder, n_features, methodology, train_x, test_x, setting)

    ratio_target = layer2_calculation(training_labels, train_y)

    def save_advanced_features(filepath, results):
        pd.DataFrame(results).to_csv(filepath, index=False)

    # Training CSV Filepath
    filepath = "{}/{}_nclusters={}_nfeatures={}_training_advanced_features.csv".format(model_folder, methodology, setting["n_clusters"], n_features)

    ratio = []
    for idx, label in enumerate(training_labels):
        ratio.append(ratio_target[label])
    results = {"ID": train_id, "Target": train_y, "Label": training_labels, "Ratio": ratio}
    save_advanced_features(filepath, results)

    # Testing CSV Filepath
    filepath = "{}/{}_nclusters={}_nfeatures={}_testing_advanced_features.csv".format(model_folder, methodology, setting["n_clusters"], n_features)

    ratio = []
    for idx, label in enumerate(testing_labels):
        ratio.append(ratio_target[label])
    results = {"ID": test_id, "Label": testing_labels, "Ratio": ratio}

    save_advanced_features(filepath, results)

def layer_aggregate_features(folder, methodology, n_clusters, N):
    advanced_training_df = None
    advanced_testing_df = None
    for idx, n_cluster in enumerate(n_clusters.split(",")):
        # kmeans_nclusters=800_nfeatures=620_training_advanced_features.csv
        filepath_training = "{}/{}_nclusters={}_nfeatures={}_training_advanced_features.csv".format(folder, methodology, n_cluster, N)
        filepath_testing = "{}/{}_nclusters={}_nfeatures={}_testing_advanced_features.csv".format(folder, methodology, n_cluster, N)

        training_df = pd.read_csv(filepath_training)
        training_df = training_df.rename(columns={"Ratio": "Ratio_nclusters={}".format(n_cluster)})
        training_df = training_df.drop(["Label"], axis=1)
        training_df.set_index(["ID"])

        testing_df = pd.read_csv(filepath_testing)
        testing_df = testing_df.rename(columns={"Ratio": "Ratio_nclusters={}".format(n_cluster)})
        testing_df = testing_df.drop(["Label"], axis=1)
        testing_df.set_index(["ID"])

        if idx == 0:
            advanced_training_df = training_df
            advanced_testing_df = testing_df
        else:
            training_df = training_df.drop(["Target"], axis=1)
            advanced_training_df = pd.concat([advanced_training_df, training_df], join="inner", axis=1)

            advanced_testing_df = pd.concat([advanced_testing_df, testing_df], join="inner", axis=1)

    size = len(n_clusters.split(","))
    advanced_training_df = advanced_training_df[np.unique(advanced_training_df.columns)]
    len_training_columns = len(advanced_training_df.columns)
    advanced_training_df = advanced_training_df.iloc[:, [idx for idx in range(size-1, len_training_columns)]]

    advanced_testing_df = advanced_testing_df[np.unique(advanced_testing_df.columns)]
    len_testing_columns = len(advanced_testing_df.columns)
    advanced_testing_df = advanced_testing_df.iloc[:, [idx for idx in range(size-1, len_testing_columns)]]

    filepath_training = "{}/{}_nclusters=all_nfeatures={}_training_advanced_features.csv".format(folder, methodology, N)
    advanced_training_df.to_csv(filepath_training, index=False)

    filepath_testing = "{}/{}_nclusters=all_nfeatures={}_testing_advanced_features.csv".format(folder, methodology, N)
    advanced_testing_df.to_csv(filepath_testing, index=False)

if __name__ == "__main__":
    from load import data_load, data_transform_2

    filepath_training = "../input/train.csv"
    filepath_testing = "../input/test.csv"

    train_x, test_x, train_y, id_train, id_test = data_transform_2(filepath_training, filepath_testing, keep_nan=True)
    #train_x, train_y, test_x, test_id = data_load()

    '''
    d = {"x": pd.Series([0, 0, 1, 1]),
         "y": pd.Series([0, 1, 0, 1])}
    train_x = pd.DataFrame(d)

    d = {"target": pd.Series([0, 1, 1, 0])}
    train_y = pd.DataFrame(d)
    train_y = train_y["target"].values
    '''

    binsize = int(sys.argv[1])
    results_single, results_couple = interaction_information(train_x, train_y, binsize=binsize, threshold=0.01)
    save_cache((results_single, results_couple), "interaction_information_transform2_binsize={}.pkl".format(binsize))

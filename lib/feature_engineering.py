#!/usr/bin/env python

import os
import sys
import time

import numpy as np
import pandas as pd

from load import load_cache, save_cache
from utils import log, INFO, WARN
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, AgglomerativeClustering, SpectralClustering
from sklearn.neighbors import NearestCentroid, kneighbors_graph
from sklearn.preprocessing import MinMaxScaler, Imputer, StandardScaler

BASEPATH = os.path.dirname(os.path.abspath(__file__))

class ClusterFactory(object):
    @staticmethod
    def get_model(method, setting):
        if method.find("kmeans") > -1:
            return Cluster(method, KMeans(n_clusters=setting["n_clusters"], n_init=setting.get("n_init", 10), init="k-means++", random_state=1201, n_jobs=setting.get("n_jobs", 1)))
        elif method.find("knn") > -1:
            return Cluster(method, NearestCentroid(shrink_threshold=setting.get("shrink", 0.1)))
        elif method.find("affinitypropagation") > -1:
            return Cluster(method, AffinityPropagation(convergence_iter=setting.get("convergence_iter", 100), max_iter=setting.get("max_iter", 5000), verbose=1))
        elif method.find("spectral") > -1:
            return Cluster(method, SpectralClustering(n_clusters=setting["n_clusters"], eigen_solver='amg', affinity="rbf"))
        elif method.find("agglomerative") > -1:
            # connectivity matrix for structured Ward
            connectivity = kneighbors_graph(setting["X"], n_neighbors=setting.get("n_neighbors", 10), include_self=False)
            # make connectivity symmetric
            connectivity = 0.5 * (connectivity + connectivity.T)

            return Cluster(method, AgglomerativeClustering(linkage="average", affinity="cityblock", n_clusters=setting["n_clusters"], connectivity=connectivity))

class Cluster(object):
    def __init__(self, name, model):
        self.name = name
        self.model = model

    def fit(self, train_x, train_y=None, is_norm=True):
        # Normalization
        if is_norm:
            train_x_min = train_x.min(0)
            train_x_ptp = train_x.ptp(axis=0)

            train_x = train_x.astype(float) - train_x_min / train_x_ptp

            if np.any(train_y):
                train_y = train_y.astype(float) - train_x_min / train_x_ptp

        imp = Imputer(missing_values='NaN', strategy='mean', axis=1)
        imp.fit(train_x)
        if np.isnan(train_x).any():
            log("Found {} NaN values in train_x, so try to transform them to 'mean'".format(np.isnan(train_x).sum()), WARN)
            train_x = imp.transform(train_x)

        if np.any(train_y) and np.isnan(train_y).any():
            log("Found {} NaN values in train_y, so try to transform them to 'mean'".format(np.isnan(train_y).sum()), WARN)
            train_y = imp.transform(train_y)

        if np.any(train_y):
            self.model.fit(train_x, train_y)
        else:
            self.model.fit(train_x)

    def predict(self, data):
        return self.model.predict(data)

    def get_centroid(self):
        return self.model.cluster_centers_

    def get_labels(self):
        return self.model.labels_

def layer1_tag_labels(model_folder, n_features, methodology, train_x, test_x, setting={}):
    timestamp_start = time.time()
    cluster_model = None
    filepath_model = "{}/{}_nclusters={}_nfeature={}.cache".format(model_folder, methodology, setting["n_clusters"], n_features)
    if os.path.exists(filepath_model):
        model = load_cache(filepath_model)

        # Backward Support
        if type(model).__name__ == "KaggleKMeans":
            cluster_model = Cluster(methodology, model.model)

            os.rename(filepath_model, "{}.kaggle".format(filepath_model))
            save_cache(model.model, filepath_model)
        else:
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

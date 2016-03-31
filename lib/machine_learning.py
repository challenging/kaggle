#!/usr/bin/env python

import os
import sys

import numpy as np

from utils import log, INFO, WARN
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, Imputer

class KaggleKMeans(object):
    def __init__(self, n_clusters, n_init=10):
        self.model = KMeans(n_clusters=n_clusters, n_init=n_init, init="random", random_state=1201)

    def fit(self, train_x):
        # Normalization
        train_x = train_x.astype(float) - train_x.min(0) / train_x.ptp(axis=0)
        print train_x.shape
        if np.isnan(train_x).any():
            log("Found {} NaN values, so try to transform them to 'mean'".format(np.isnan(train_x).sum()), WARN)

            imp = Imputer(missing_values='NaN', strategy='mean', axis=1)
            imp.fit(train_x)

            train_x = imp.transform(train_x)

        self.model.fit(train_x)

    def get_centroid(self):
        return self.model.cluster_centers_

    def get_labels(self):
        return self.model.labels_

    def stats(self, train_y):
        pass

if __name__ == "__main__":
    BASEPATH = "."
    n_clusters = 800

    import time

    from load import pca, data_load, data_transform_2, load_cache, save_cache
    drop_fields = ['v8','v23','v25','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128']
    N = 650 - len(drop_fields)

    filepath_training = "{}/../input/train.csv".format(BASEPATH)
    filepath_testing = "{}/../input/test.csv".format(BASEPATH)

    filepath_cache_1 = "../input/{}_training_dataset.cache".format(N)
    model_folder = "../prediction_model/others"

    train_x, test_x, train_y, test_id = None, None, None, None
    if os.path.exists(filepath_cache_1):
        train_x, test_x, train_y, test_id, train_id = load_cache(filepath_cache_1)

        log("Load data from cache file({}) for the original data sources".format(filepath_cache_1), INFO)
    else:
        train_x, test_x, train_y, test_id = data_transform_2(filepath_training, filepath_testing, drop_fields)

        save_cache((train_x, test_x, train_y, test_id), filepath_cache_1)
        log("Save data to cache file({}) for the original data sources".format(filepath_cache_1))

    train_X, test_X = train_x.values, test_x.values
    train_y, test_id = train_y.values, test_id.values
    train_Y = train_y.astype(float)

    timestamp_start = time.time()
    layer_1_cluster = None
    filepath_model = "{}/kmeans_nclusters={}_nfeature={}.cache".format(model_folder, n_clusters, N)
    if os.path.exists(filepath_model):
        layer_1_cluster = load_cache(filepath_model)

        print layer_1_cluster.__class__
        if layer_1_cluster.__class__.__name__ == "KaggleKMeans":
            os.rename(filepath_model, "{}.kaggle".format(filepath_model))

            save_cache(layer_1_cluster.model, filepath_model)
    else:
        layer_1_cluster = KaggleKMeans(n_clusters)
        layer_1_cluster.fit(train_X)

        save_cache(layer_1_cluster, filepath_model)

    print "Cost {} secends to build KMeans model".format(time.time() - timestamp_start)

    '''
    for centroid in layer_1_cluster.get_centroid():
        print centroid
    '''

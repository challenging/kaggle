#!/usr/bin/env python

import os
import sys

from utils import log, INFO
from sklearn.cluster import KMeans

class KaggleKMeans(object):
    def __init__(self, n_clusters, n_init=10):
        self.model = KMeans(n_clusters=n_clusters, n_init=n_init, init="random", random_state=1201)

    def fit(self, train_x):
        self.model.fit(train_x)

    def get_centroid(self):
        return self.model.cluster_centers_

    def get_labels(self):
        return self.model.labels_

    def stats(self, train_y):
        pass

if __name__ == "__main__":
    BASEPATH = "."
    n_clusters = 200

    import time

    from load import pca, data_load, data_transform_2, load_cache, save_cache
    N = 650

    filepath_training = "{}/../input/train.csv".format(BASEPATH)
    filepath_testing = "{}/../input/test.csv".format(BASEPATH)

    filepath_cache_1 = "../input/{}_training_dataset.cache".format(N)
    model_folder = "../prediction_model/others"

    train_x, test_x, train_y, test_id = None, None, None, None
    if os.path.exists(filepath_cache_1):
        train_x, test_x, train_y, test_id = load_cache(filepath_cache_1)

        log("Load data from cache file({}) for the original data sources".format(filepath_cache_1), INFO)
    else:
        train_x, test_x, train_y, test_id = data_transform_2(filepath_training, filepath_testing)

        save_cache((train_x, test_x, train_y, test_id), filepath_cache_1)
        log("Save data to cache file({}) for the original data sources".format(filepath_cache_1))

    train_X, test_X = train_x.values, test_x.values
    train_y, test_id = train_y.values, test_id.values
    train_Y = train_y.astype(float)

    timestamp_start = time.time()
    layer_1_cluster = None
    filepath_model = "{}/kmeans_{}.cache".format(model_folder, n_clusters)
    if os.path.exists(filepath_model):
        layer_1_cluster = load_cache(filepath_model)
    else:
        layer_1_cluster = KaggleKMeans(n_clusters)
        layer_1_cluster.fit(train_X)
    print "Cost {} secends to build KMeans model".format(time.time() - timestamp_start)

    for centroid in layer_1_cluster.get_centroid():
        print centroid

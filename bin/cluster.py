#!/usr/bin/env python

import os
import sys

import click
import numpy as np

sys.path.append("{}/../lib".format(os.path.dirname(os.path.abspath(__file__))))
from utils import log, INFO
from load import load_data
from cluster_process import layer_process, layer_aggregate_features

BASEPATH = os.path.dirname(os.path.abspath(__file__))

@click.command()
@click.option("--methodology", required=True, help="Which clustering methodology")
@click.option("--n-clusters", default="100", help="Number of clusters")
def cluster(methodology, n_clusters):
    drop_fields = []
    drop_fields = ['v8','v23','v25','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128']

    N = 650 - len(drop_fields)

    filepath_training = "{}/../input/train.csv".format(BASEPATH)
    filepath_testing = "{}/../input/test.csv".format(BASEPATH)
    filepath_cache_1 = "../input/{}_training_dataset.cache".format(N)

    model_folder = "{}/../prediction_model/others".format(BASEPATH)

    train_x, test_x, train_y, test_id, train_id = load_data(filepath_cache_1, filepath_training, filepath_testing, drop_fields)

    train_X, test_X = train_x.values, test_x.values
    train_y = train_y.values
    test_id = test_id.values
    train_id = train_id.values
    train_Y = train_y.astype(float)

    for n_cluster in n_clusters.split(","):
        log("Start to build {} model for {} n_cluster".format(methodology, n_cluster), INFO)
        setting = {"n_clusters": int(n_cluster),
                   "n_jobs": -1,
                   "n_init": 10}

        if methodology.find("agglomerative") > -1:
            setting["X"] = train_X
            setting["n_neighbors"] = 20

            layer_process(model_folder, N, methodology, train_X, train_y, train_id, test_X, test_id, setting=setting)

    layer_aggregate_features(model_folder, methodology, n_clusters, N)

if __name__ == "__main__":
    cluster()

#!/usr/bin/env python

import os
import sys
sys.path.append("{}/../lib".format(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import click

from load import *
from deep_learning import lr

BASEPATH = os.path.dirname(os.path.abspath(__file__))

@click.command()
@click.option("--pca-number", default=None, help="Dimension of PCA")
@click.option("--mini-batch", default=5, help="Number of mini batch")
@click.option("--dimension", default=100, help="Dimension of Hiddle Layer")
@click.option("--layer", default=2, help="Number of Hidden Layer")
@click.option("--ratio", default=1, help="Ratio for sample balance")
@click.option("--iteration", default=10, help="Run x iteration for training")
@click.option("--learning-rate", default=1e-6, help="Learning Rate for model")
@click.option("--transform", default=1, help="Tranform Methodology")
def main(pca_number, mini_batch, dimension, layer, ratio, iteration, learning_rate, transform):
    train_x, train_y, test_x, test_id = data_load(drop_fields=['v8','v23','v25','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'])

    #train_x, train_y, test_x, test_id = data_load()

    if transform == 1:
        train_x, test_x = data_transform_1(train_x, test_x)
    elif transform == 2:
        train_x, test_x = data_transform_2(train_x, test_x)
    else:
        click.echo("Not Found the tranform metholody {}".format(transform))
        sys.exit(1)

    train_X, test_X = train_x.values, test_x
    pca_model = pca(train_X, int(pca_number) if pca_number else None)
    train_pca_X = pca_model.fit_transform(train_X)
    test_pca_X = pca_model.fit_transform(test_x)

    train_Y = train_y.astype(float)

    number_of_feature = len(train_pca_X[0])
    print "Data Distribution is ({}, {}), and then the number of feature is {}".format(np.sum(train_Y==0), np.sum(train_Y==1), number_of_feature),

    model_folder = "{}/../prediction_model/deep_learning/transform={}/learning_rate={}//mini_batch={}/feature={}/layer={}/dimension={}".format(\
                        BASEPATH, transform, learning_rate, mini_batch, number_of_feature, layer, dimension)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    print "prepare to save data in {}".format(model_folder)

    model = lr(model_folder, layer, mini_batch, dimension,
               train_pca_X, train_Y,
               testing_data=test_pca_X, testing_id=test_id,
               nepoch=iteration, class_weight={0: ratio, 1: 1}, learning_rate=learning_rate)

if __name__ == "__main__":
    main()

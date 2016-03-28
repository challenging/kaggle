#!/usr/bin/env python

import os
import sys

import click
import numpy as np

sys.path.append("{}/../lib".format(os.path.dirname(os.path.abspath(__file__))))
from load import data_load, data_transform_1, data_transform_2, save_kaggle_submission, pca
from learning import LearningFactory
from ensemble_learning import ensemble_model

BASEPATH = os.path.dirname(os.path.abspath(__file__))

@click.command()
@click.option("--nfold", default=10, help="Number of fold")
@click.option("--estimators", default=100, help="Number of estimator")
@click.option("--thread", default=1, help="Number of thread")
@click.option("--pca-number", default=None, help="Dimension of PCA")
@click.option("--transform", default=1, help="Tranform Methodology")
def learning(pca_number, transform, thread, nfold, estimators):
    drop_fields = ['v8','v23','v25','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128']
    train_x, train_y, test_x, test_id = data_load(drop_fields=drop_fields, filepath_cache="../input/455_dataset.cache")

    if transform == 1:
        train_x, test_x = data_transform_1(train_x, test_x)
    elif transform == 2:
        filepath = "{}/../input/transform2.csv".format(BASEPATH)
        train_x, test_x = data_transform_2(train_x, test_x, filepath)
    else:
        click.echo("Not Found the tranform metholody {}".format(transform))
        sys.exit(1)

    # Init the parameters
    LearningFactory.set_n_estimators(estimators)

    train_X = train_x.values
    if pca_number == 0:
        train_pca_X = train_X
        test_pac_X = test_x
    else:
        pca_model = pca(train_X, int(pca_number) if pca_number else None)
        train_pca_X = pca_model.fit_transform(train_X)
        test_pca_X = pca_model.fit_transform(test_x)

    train_Y = train_y.astype(float)

    number_of_feature = len(train_pca_X[0])
    if not pca_number:
        pca_number = number_of_feature

    model_folder = "{}/../prediction_model/ensemble_learning/transform={}_feature={}_pcanumber={}_estimators={}".format(\
                        BASEPATH, transform, number_of_feature, pca_number, estimators)

    print "Data Distribution is ({}, {}), and then the number of feature is {}".format(np.sum(train_Y==0), np.sum(train_Y==1), number_of_feature),

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    print "prepare to save data in {}".format(model_folder)

    models = ["shadow_extratree_regressor", "shadow_extratree_classifier",
              "shadow_randomforest_regressor", "shadow_randomforest_classifier",
              #"shadow_xgboosting_regressor", # The logloss value is always nan, why???
              "shadow_xgboosting_classifier",
              "shadow_gradientboosting_regressor", "shadow_gradientboosting_classifier"]
    results = ensemble_model(model_folder, train_pca_X, train_Y, test_pca_X, test_id, models, n_folds=nfold, number_of_thread=thread,
                             filepath_queue="{}/queue.pickle".format(model_folder), filepath_nfold="{}/nfold.pickle".format(model_folder))

    filepath_output = "{}/kaggle_BNP_submission.csv".format(model_folder)
    save_kaggle_submission(test_id, results, filepath_output)

if __name__ == "__main__":
    learning()

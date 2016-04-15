#!/usr/bin/env python

import os
import sys
import time
import pickle
import operator

import numpy as np
import pandas as pd

from utils import log, DEBUG, INFO, WARN
from sklearn import decomposition
from sklearn.preprocessing import PolynomialFeatures

def data_load(filepath_train="../input/train.csv", filepath_test="../input/test.csv", drop_fields=[], filepath_cache=None):
    log("Load data...", INFO)

    train_x, train_y, test_x, test_id = None, None, None, None
    if filepath_cache and os.path.exists(filepath_cache):
        with open(filepath_cache, "rb") as INPUT:
            train_x, train_y, test_x, test_id = pickle.load(INPUT)

        log("Load data from cache, {}".format(filepath_cache))
    else:
        log("Start to load data from {}".format(filepath_train))

        test_x = pd.read_csv(filepath_test)
        test_id = test_x["ID"].values
        drop_fields.extend(["ID"])
        test_x = test_x.drop(drop_fields, axis=1)

        train = pd.read_csv(filepath_train)
        train_y = train['target'].values
        drop_fields.extend(["target"])
        train_x = train.drop(drop_fields, axis=1)

        if filepath_cache:
            with open(filepath_cache, "wb") as OUTPUT:
                pickle.dump((train_x, train_y, test_x, test_id), OUTPUT)

    return train_x, train_y, test_x, test_id

def pca(x, number_of_feature=None):
    pca = decomposition.PCA(n_components=number_of_feature)
    pca.fit(x)

    return pca

def data_transform_1(train, test):
    for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems()):
        if train_series.dtype == 'O':
            #for objects: factorize
            train[train_name], tmp_indexer = pd.factorize(train[train_name])
            test[test_name] = tmp_indexer.get_indexer(test[test_name])
            #but now we have -1 values (NaN)
        else:
            #for int or float: fill NaN
            tmp_len = len(train[train_series.isnull()])
            if tmp_len > 0:
                #print "mean", train_series.mean()
                train.loc[train_series.isnull(), train_name] = -999

            #and Test
            tmp_len = len(test[test_series.isnull()])
            if tmp_len>0:
                test.loc[test_series.isnull(), test_name] = -999

    return train, test

def data_transform_2(filepath_training, filepath_testing, drop_fields=[], keep_nan=False):
    log("Try to load CSV files, {} and {}".format(filepath_training, filepath_testing), INFO)

    train = pd.read_csv(filepath_training)
    test = pd.read_csv(filepath_testing)

    if drop_fields:
        train = train.drop(drop_fields, axis=1)
        test = test.drop(drop_fields, axis=1)

    num_train = train.shape[0]

    y_train = train['target']
    train = train.drop(['target'], axis=1)
    id_test = test['ID']

    def fill_nan_null(val):
        ret_fill_nan_null = 0.0
        if val == True:
            ret_fill_nan_null = 1.0

        return ret_fill_nan_null

    id_train = train["ID"]

    df_all = pd.concat((train, test), axis=0, ignore_index=True)
    df_all['null_count'] = df_all.isnull().sum(axis=1).tolist()

    df_all_temp = df_all['ID']

    df_all = df_all.drop(['ID'],axis=1)
    df_data_types = df_all.dtypes[:] #{'object':0,'int64':0,'float64':0,'datetime64':0}
    d_col_drops = []

    for i in range(len(df_data_types)):
        key = str(df_data_types.index[i])+'_nan_'
        tmp_column = df_all[str(df_data_types.index[i])].map(lambda x:fill_nan_null(pd.isnull(x)))

        if len(tmp_column.unique()) > 1:
            df_all[key] = tmp_column

    if not keep_nan:
        df_all = df_all.fillna(-9999)

    log("Try to convert 'categorical variable to onehot vector'", INFO)
    for i in range(len(df_data_types)):
        if str(df_data_types[i]) == 'object':
            df_u = pd.unique(df_all[str(df_data_types.index[i])].ravel())

            d = {}
            j = 1000
            for s in df_u:
                d[str(s)] = j
                j += 5
            df_all[str(df_data_types.index[i])+'_vect_'] = df_all[str(df_data_types.index[i])].map(lambda x:d[str(x)])
            d_col_drops.append(str(df_data_types.index[i]))

            if len(df_u) < 150:
                dummies = pd.get_dummies(df_all[str(df_data_types.index[i])]).rename(columns=lambda x: str(df_data_types.index[i]) + '_' + str(x))
                df_all_temp = pd.concat([df_all_temp, dummies], axis=1)

    if isinstance(df_all_temp, pd.DataFrame):
        df_all_temp = df_all_temp.drop(['ID'],axis=1)
        df_all = pd.concat([df_all, df_all_temp], axis=1)

    train = df_all.iloc[:num_train]
    test = df_all.iloc[num_train:]
    train = train.drop(d_col_drops,axis=1)
    test = test.drop(d_col_drops,axis=1)
    log("Finish the whole data process", INFO)

    return train, test, y_train, id_test, id_train

def data_polynomial(filepath, train_x, train_y):
    def polynomial(dataset):
        timestamp_start = time.time()
        log("Start to feature extending by polynomial for training dataset", INFO)
        dataset = PolynomialFeatures(interaction_only=True).fit_transform(dataset)
        log("Cost {} secends to finish".format(time.time() - timestamp_start), INFO)

        return dataset

    if os.path.exists(filepath):
        train_x, test_x = load_cache(filepath)
        log("Load cache file from {}".format(filepath), INFO)
    else:
        train_x = polynomial(train_x)[1:]
        test_x = polynomial(train_y)[1:]

        save_cache((train_x, test_x), filepath)
        log("Save cache in {}".format(filepath), INFO)

    return train_x, test_x

def load_data(filepath, filepath_training, filepath_testing, drop_fields=[]):
    train_x, test_x, train_y, test_id, train_id = None, None, None, None, None
    if os.path.exists(filepath):
        train_x, test_x, train_y, test_id, train_id = load_cache(filepath)
    else:
        train_x, test_x, train_y, test_id, train_id = data_transform_2(filepath_training, filepath_testing, drop_fields)
        save_cache((train_x, test_x, train_y, test_id, train_id), filepath)

    return train_x, test_x, train_y, test_id, train_id

def load_advanced_data(filepath_training, filepath_testing, drop_fields=[]):
    if os.path.exists(filepath_training) and os.path.exists(filepath_testing):
        df_train = pd.read_csv(filepath_training)
        df_train = df_train.drop(["Target"], axis=1)
        df_train = df_train.drop(drop_fields, axis=1)

        df_test = pd.read_csv(filepath_testing)
        df_test = df_test.drop(drop_fields, axis=1)
    else:
        log("Not Found {} or {}".format(filepath_training, filepath_testing), INFO)
        return None, None

    return df_train, df_test

def load_interaction_information(filepath, count=None, threshold=None):
    results = None
    with open(filepath, "rb") as INPUT:
        results = pickle.load(INPUT)

    matching_type = (count != None)
    ranking = {}
    for layer1, info in results.items():
        if isinstance(info, dict):
            for layer2, value in info.items():
                ranking["{}-{}".format(layer1, layer2)] = value
        elif isinstance(info, float):
            layer1 = layer1.replace(";", "-")

            if count:
                ranking[layer1] = info
            elif threshold and info > threshold:
                fields = layer1.split("-")
                if "target" in fields:
                    fields.remove("target")

                yield fields, info

    if matching_type:
        for key, value in sorted(ranking.items(), key=operator.itemgetter(1), reverse=True):
            fields = key.split("-")
            if "target" in fields:
                fields.remove("target")

            yield fields, value

            if count < 2:
                break
            else:
                count -= 1

def save_kaggle_submission(test_id, results, filepath, normalization=False):
    if normalization:
        results = (results - results.min()) / (results.max() - results.min())

    pd.DataFrame({"ID": test_id, "PredictedProb": results}).to_csv(filepath, index=False)

def save_cache(obj, filepath):
    parent_folder = os.path.dirname(filepath)
    if not os.path.isdir(parent_folder):
        os.makedirs(parent_folder)

    with open(filepath, "wb") as OUTPUT:
        pickle.dump(obj, OUTPUT)

    log("Save {}'s cache in {}".format(obj.__class__, filepath), DEBUG)

def load_cache(filepath):
    log("Try to load {}".format(filepath))

    obj = None
    try:
        with open(filepath, "rb") as INPUT:
            obj = pickle.load(INPUT)

        log("Load {} from cache, {}".format(obj.__class__, filepath), INFO)
    except ValueError as e:
        log("{} when loading pickle file so removing {}".format(str(e), filepath), WARN)

        os.remove(filepath)
        sys.exit(100)

    return obj

if __name__ == "__main__":
    drop_fields = []
    BASEPATH = "/Users/RungChiChen/Documents/kaggle/Santander Customer Satisfaction"

    filepath_training = "{}/input/train.csv".format(BASEPATH)
    filepath_testing = "{}/input/test.csv".format(BASEPATH)
    filepath_cache_1 = "{}/input/training_dataset.cache".format(BASEPATH)

    train_x, test_x, train_y, test_id, train_id = load_data(filepath_cache_1, filepath_training, filepath_testing, drop_fields)

    filepath_interaction_information = "{}/input/transform2=True_testing=-1_type=2_binsize=4_combination=2.pkl".format(BASEPATH)
    for layers, value in load_interaction_information(filepath_interaction_information):
        print layers, value

        break

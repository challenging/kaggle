#!/usr/bin/env python

import os
import sys
import pickle

import numpy
import pandas as pd

from utils import log, INFO, WARN
from sklearn import decomposition

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

def data_transform_2(filepath_training, filepath_testing, drop_fields=[]):
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

    df_all = pd.concat((train, test), axis=0, ignore_index=True)
    df_all['null_count'] = df_all.isnull().sum(axis=1).tolist()
    df_all_temp = df_all['ID']
    df_all = df_all.drop(['ID'],axis=1)
    df_data_types = df_all.dtypes[:] #{'object':0,'int64':0,'float64':0,'datetime64':0}
    d_col_drops = []

    for i in range(len(df_data_types)):
        df_all[str(df_data_types.index[i])+'_nan_'] = df_all[str(df_data_types.index[i])].map(lambda x:fill_nan_null(pd.isnull(x)))
    df_all = df_all.fillna(-9999)

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

    df_all_temp = df_all_temp.drop(['ID'],axis=1)
    df_all = pd.concat([df_all, df_all_temp], axis=1)

    train = df_all.iloc[:num_train]
    test = df_all.iloc[num_train:]
    train = train.drop(d_col_drops,axis=1)
    test = test.drop(d_col_drops,axis=1)

    return train, test, y_train, id_test

def data_balance(x, y, criteria, ratio):
    print "Balance data by ratio={}".format(ratio)

    idxs = numpy.where(y==criteria)[0]

    x_set = x[idxs]
    y_set = y[idxs]
    for i in range(0, int(ratio)-1):
        x = numpy.concatenate((x, x_set), axis=0)
        y = numpy.concatenate((y, y_set), axis=0)

    if ratio - int(ratio) > 0:
        idxs = idxs[:int(len(idxs)*(ratio-int(ratio)))]
        x = numpy.concatenate((x, x[idxs]), axis=0)
        y = numpy.concatenate((y, y[idxs]), axis=0)

    return x, y

def save_kaggle_submission(test_id, results, filepath, normalization=False):
    if normalization:
        results = (results - results.min()) / (results.max() - results.min())

    pd.DataFrame({"ID": test_id, "PredictedProb": results}).to_csv(filepath, index=False)

def save_cache(obj, filepath):
    with open(filepath, "wb") as OUTPUT:
        pickle.dump(obj, OUTPUT)

    log("Save {}'s cache in {}".format(obj.__class__, filepath), INFO)

def load_cache(filepath):
    log("Try to load {}".format(filepath))

    obj = None
    try:
        with open(filepath, "rb") as INPUT:
            obj = pickle.load(INPUT)

        log("Load {} from cache, {}".format(obj.__class__, filepath), INFO)
    except ValueError as e:
        log("Error when loading pickle file so removing it", WARN)

        os.remove(filepath)
        sys.exit(100)

    return obj

if __name__ == "__main__":
    train_x, train_y, test_x, test_id = data_load()

    train_X, test_X = data_transform_1(train_x, test_x)
    print train_X.columns
    print train_X.values[0]

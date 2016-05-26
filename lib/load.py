#:!/usr/bin/env python

import os
import sys
import glob
import time
import pickle
import operator

import json

import numpy as np
import pandas as pd

from bz2 import BZ2File
from bsddb3 import db
from memory_profiler import profile
from utils import log, DEBUG, INFO, WARN, create_folder
from sklearn.preprocessing import PolynomialFeatures

MONGODB_URL = "mongodb://127.0.0.1:27017"
MONGODB_DATABASE = "kaggle_cache"
MONGODB_BATCHSIZE = 10000

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

    id_train = None
    if "ID" in train.columns:
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

def load_interaction_information(folder, threshold=300, reverse=True):
    SPLIT_SYMBOL = ";"

    results = {}
    for filepath in glob.glob("{}/*pkl".format(folder)):
        with open(filepath, "rb") as INPUT:
            results.update(pickle.load(INPUT))

    is_integer = threshold.isdigit()
    if is_integer:
        threshold = int(threshold)
    else:
        threshold = float(threshold)

    for_integer = {}
    ranking = {}
    for layer1, score in results.items():
        fields = layer1.split(SPLIT_SYMBOL)
        if "target" in fields:
            fields.remove("target")

        if len(fields) == 1:
            continue

        if is_integer:
            size = len(fields)
            for_integer[size] = threshold

            ranking[SPLIT_SYMBOL.join(fields)] = score
        elif score > threshold:
            yield fields, score

    if is_integer:
        for key, value in sorted(ranking.items(), key=operator.itemgetter(1), reverse=reverse):
            fields = key.split(SPLIT_SYMBOL)
            size = len(fields)
            for_integer[size] -= 1

            if for_integer[size] >= 0:
                yield fields, value

            all_zero = True
            for t, value in for_integer.items():
                if value >= 0:
                    all_zero = False

                    break

            if all_zero:
                break

def load_feature_importance(filepath_pkl, top=512):
    ranking = load_cache(filepath_pkl)
    records = ranking["Mean"]

    columns = []
    for name, score in sorted(records.items(), key=operator.itemgetter(1), reverse=True):
        columns.append(name)
        if len(columns) >= top:
            break

    return columns

def save_kaggle_submission(results, filepath):
    pd.DataFrame(results).to_csv(filepath, index=False)

def save_cache(obj, filepath, is_json=False, is_hdb=False):
    if is_hdb:
        filepath += ".hdb"

        hdb = db.DB()
        hdb.open(filepath, None, db.DB_HASH, db.DB_CREATE)

        for test_id, info in obj.items():
            hdb.put(str(test_id), pickle.dumps(info))

        hdb.sync()
        hdb.close()

        log("Save cache in BerkeleyDB format({})".format(filepath), INFO)
    elif is_json:
        filepath += ".json.bz2"

        with BZ2File(filepath, "wb") as OUTPUT:
            json.dump(obj, OUTPUT)

        log("Save cache in JSON format({})".format(filepath), INFO)
    else:
        create_folder(filepath)
        with open(filepath, "wb") as OUTPUT:
            pickle.dump(obj, OUTPUT)

        log("Save {}'s cache in {}".format(obj.__class__, filepath), INFO)

def load_cache(filepath, is_json=False, is_hdb=False, others=None):
    obj = {}
    weight = 1

    timestamp_start = time.time()
    if is_hdb:
        if others:
            obj, weight = others

        filepath += ".hdb"

        hdb = db.DB()
        hdb.open(filepath, None, db.DB_HASH, db.DB_CREATE)

        cursor = hdb.cursor()
        rec = cursor.first()
        while rec:
            test_id, value = int(rec[0]), pickle.loads(rec[1])
            obj.setdefault(test_id, {})

            for place_id, score in value.items():
                place_id = int(float(place_id))

                obj[test_id].setdefault(place_id, 0)
                obj[test_id][place_id] += score*weight

                if test_id == "7":
                    log("{} -> {},{} -> {}".format(place_id, score, weight, obj[test_id][place_id]))

            rec = cursor.next()

        hdb.close()
    elif is_json:
        filepath += ".json.bz2"

        if os.path.exists(filepath):
            with BZ2File(filepath, "rb") as INPUT:
                obj = json.load(INPUT).items()
        else:
            log("Not found file in {}".format(filepath), WARN)
    else:
        try:
            if os.path.exists(filepath):
                with open(filepath, "rb") as INPUT:
                    obj = pickle.load(INPUT)
            else:
                log("Not found file in {}".format(filepath), WARN)
        except ValueError as e:
            log("when loading pickle file so removing {}".format(filepath), WARN)
            os.remove(filepath)
        except EOFError as e:
            log("when loading pickle file so removing {}".format(filepath), WARN)
            os.remove(filepath)
        except KeyError as e:
            log("when loading pickle file so removing {}".format(filepath), WARN)
            os.remove(filepath)
        except IndexError as e:
            log("when loading pickle file so removing {}".format(filepath), WARN)
            os.remove(filepath)

    timestamp_end = time.time()
    log("Spend {:8f} seconds to load cache from {}".format(timestamp_end-timestamp_start, filepath), INFO)

    return obj

if __name__ == "__main__":
    for test_id, info in load_cache("/Users/rongqichen/Documents/programs/kaggle/cases/Facebook V - Predicting Check Ins/input/cache/64x64_windowsize=1_batchsize=500000_isaccuracy=False_excludeoutlier=False_istesting=False/method=most_popular_54cc3480ffb5b30f7744eba8dd2f663c.6/a32d479f49d2a30e4e3b57bb494dcac6.pkl").items():
        print test_id, info
        print type(test_id).__name__
        break

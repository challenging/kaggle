# coding=UTF-8

import os
import sys
import glob
import time
import random

import re
import numpy as np
import pandas as pd

from itertools import combinations
from collections import Counter

from load import save_cache, load_cache, load_interaction_information
from interaction_information import InteractionInformation, InteractionInformationThread
from utils import log, DEBUG, INFO, WARN

def load_dataset(filepath_cache, dataset, binsize=2, threshold=0.1):
    LABELS = "abcdefghijklmnopqrstuvwxABCDEFGHIJKLMNOPQRSTUVWX0123456789!@#$%^&*()_+~"
    FIXED_LABELS = "yYz"

    def less_replace(df, c, unique_values, labels):
        for i, unique_value in enumerate(unique_values):
            df[c][df[c] == unique_value] == labels[i]

    drop_columns = []
    if os.path.exists(filepath_cache):
        dataset = load_cache(filepath_cache)
    else:
        count_raw = len(dataset[dataset.columns[0]].values)
        for idx, column in enumerate(dataset.columns):
            data_type = dataset.dtypes[idx]
            unique_values = dataset[column].unique()

            try:
                if column != "target":
                    if data_type == "object":
                        if len(unique_values) < len(LABELS):
                            less_replace(dataset, column, unique_values, LABELS)
                            log("Change {} by unique type(size={})".format(column, len(unique_values)), INFO)
                        else:
                            log("The size of {} is too large({})".format(column, len(unique_values)), WARN)
                    else:
                        is_break = False
                        deleted_idxs = np.array([])
                        counter = Counter(dataset[column].values).most_common(len(FIXED_LABELS))
                        for idx_label, (name, value) in enumerate(counter):
                            ratio = float(value) / count_raw

                            if ratio == 1:
                                drop_columns.append(column)
                                log("The size of most common value of {} is 1 so skipping it".format(column), INFO)

                                is_break = True
                                break
                            elif ratio > threshold:
                                log("The ratio of common value({}, {}) of {} is {}, greater".format(data_type, name, column, ratio), INFO)

                                idxs_most_common = np.where(dataset[column] == name)[0]
                                deleted_idxs = np.concatenate((deleted_idxs, idxs_most_common), axis=0)

                                dataset[column][idxs_most_common] = FIXED_LABELS[idx_label]
                            else:
                                log("The ratio of common value({}, {}) of {} is {}, less".format(data_type, name, column, ratio), INFO)

                                break

                        if is_break:
                            continue
                        else:
                            ori_idxs = np.array([tmp_i for tmp_i in range(0, count_raw)])
                            idxs_non_most_common = np.delete(ori_idxs, deleted_idxs)

                            non_common_unique_values = dataset[column][idxs_non_most_common].unique()

                            if len(non_common_unique_values) < len(LABELS):
                                for ii, unique_value in enumerate(non_common_unique_values):
                                    dataset[column][dataset[column] == unique_value] = LABELS[ii]
                            else:
                                is_success = False
                                for tmp_binsize in [t for t in range(len(LABELS)-1, 0, -4)]:
                                    try:
                                        tmp = pd.qcut(dataset[column][idxs_non_most_common].values, tmp_binsize, labels=[c for c in LABELS[:tmp_binsize]])
                                        dataset[column][idxs_non_most_common] = tmp
                                        is_success = True

                                        break
                                    except ValueError as e:
                                        if e.message.find("Bin edges must be unique") > -1:
                                            log("Descrease binsize from {} to {} for {} again due to {}".format(column, tmp_binsize, tmp_binsize-4, str(e)), DEBUG)
                                        else:
                                            raise

                                if is_success:
                                    log("The final binsize of {} is {}".format(column, tmp_binsize), INFO)
                                else:
                                    log("Fail in transforming {}".format(column), WARN)
                                    drop_columns.append(column)

                                    continue

                            log("Change {} by bucket type".format(column), INFO)

                    dataset[column] = ["Z" if str(value) == "nan" else value for value in dataset[column]]
                else:
                    log("The type of {} is already categorical".format(column), INFO)
            except ValueError as e:
                log("The size of unique values of {} is {}, greater than {}".format(column, len(unique_values), binsize), INFO)
                raise

        dataset = dataset.drop(drop_columns, axis=1)

        dataset.to_csv("{}.csv".format(filepath_cache))
        save_cache(dataset, filepath_cache)

    return dataset

def calculate_interaction_information(filepath_cache, dataset, train_y, folder_couple, combinations_size,
                                      n_split_idx=0, n_split_num=1, binsize=2, nthread=4, is_testing=None):
    dataset = load_dataset(filepath_cache, dataset, binsize)

    ii = InteractionInformation(dataset, train_y, folder_couple, combinations_size)

    count_break = 0

    for size in range(combinations_size, 1, -1):
        rounds = list(combinations([column for column in dataset.columns], size))
        for pair_column in rounds[n_split_idx::n_split_num]:
            if is_testing and random.random()*10 > 1: # Random Sampling when is_testing = True
                continue

            ii.add_item(pair_column, size)

            if is_testing and count_break > is_testing:
                log("Early break due to the is_testing is True", INFO)
                break
            else:
                count_break += 1

    # Memory Concern
    ii.results_couple = {}

    threads = []
    for idx in range(0, nthread):
        worker = InteractionInformationThread(kwargs={"ii": ii, "results_couple": {}, "folder_couple": folder_couple, "batch_size_dump": 2**13})
        worker.setDaemon(True)
        worker.start()

        threads.append(worker)

    log("Wait for the completion of the calculation of Interaction Information", INFO)
    ii.queue.join()

    # Force dumpping the results
    for t in threads:
        t.dump(True)

    return ii.results_couple

def merge_interaction_information(folder_couple, size_dump=2**15):
    def save(obj):
        filepath = "{}/{}.merge.pkl".format(folder_couple, int(100000*time.time()))
        save_cache(obj, filepath)

        return 1

    results = {}
    count_filepath, count_couple, final_count_filepath, final_count_couple = 0, 0, 0, 0

    for filepath in glob.iglob("{}/*pkl".format(folder_couple)):
        with open(filepath, "rb") as INPUT:
            o = load_cache(filepath)
            results.update(o)

            count_couple += len(o)

        os.rename(filepath, "{}.bak".format(filepath))
        count_filepath += 1

    if count_filepath == 0:
        log("Not file in {}".format(folder_couple), WARN)
    else:
        dump_results = {}
        for idx, key in enumerate(results.keys()):
            dump_results[key] = results[key]
            final_count_couple += 1

            if len(dump_results) > size_dump:
                final_count_filepath += save(dump_results)
                dump_results = {}

        final_count_filepath += save(dump_results)

    return count_filepath, count_couple, final_count_filepath, final_count_couple

def test_new_interaction_information(filepath_cache, dataset, train_y, binsize=4):
    import math
    from information_discrete import mi

    dataset = load_dataset(filepath_cache, dataset, binsize)

    # ind_var34;saldo_medio_var5_ult3;target 0.0149167532518
    a = mi(dataset["ind_var34"].values, dataset["saldo_medio_var5_ult3"].values)
    b = mi_3d(("ind_var34", dataset["ind_var34"].values), ("saldo_medio_var5_ult3", dataset["saldo_medio_var5_ult3"].values), ("target", train_y.values))

    distribution = {}
    x = np.unique(dataset["ind_var34"].values)
    y = np.unique(dataset["saldo_medio_var5_ult3"].values)
    z = np.unique(train_y.values)

    from entropy_estimators import cmidd
    c = cmidd(dataset["ind_var34"].values, dataset["saldo_medio_var5_ult3"].values, train_y.values)

    print c, a, b, c-a

    print mi_4d(("var3", dataset["var3"].values), ("ind_var34", dataset["ind_var34"].values), ("saldo_medio_var5_ult3", dataset["saldo_medio_var5_ult3"].values), ("target", train_y.values))

def merge_binsize(filepath_output, pattern, topX=500):
    dfs = []

    for filepath in glob.glob(pattern):
        if filepath.find("cache") == -1:
            filename = os.path.basename(filepath)
            binsize = re.search("(binsize=(\d+))", filename).groups()[0]

            index, series = [], {"{}_value".format(binsize): [], "{}_rank".format(binsize): []}
            for key, value in load_interaction_information(filepath, topX):
                index.append(";".join(key))
                series["{}_value".format(binsize)].append(value)
                series["{}_rank".format(binsize)].append(len(index))

            dfs.append(pd.DataFrame(series, index=index))

    # Merge
    results = pd.concat(dfs, axis=1)
    results.to_csv(filepath_output)

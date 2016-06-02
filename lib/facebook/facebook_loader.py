#!/usr/bin/env python

import os
import copy
import operator

import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from utils import create_folder, log, INFO

def _complex_split_data(df, time_column, time_id, range_x, range_y, size_x, size_y, output_folder):
    c1 = (df[time_column] == time_id)

    folder = os.path.join(output_folder, "{}={}".format(time_column, time_id))
    for window_size_x, window_size_y in zip(size_x, size_y):
        for x in range_x:
            start_x, end_x = x, min(11, x+window_size_x)
            c2 = (df["x"].values >= start_x) & (df["x"].values < end_x)

            for y in range_y:
                start_y, end_y = y, min(11, y+window_size_y)
                c3 = (df["y"].values >= start_y) & (df["y"].values < end_y)

                filepath_output = os.path.join(folder, "windown_size={},{}".format(window_size_x, window_size_y), "{}_{}.csv".format(start_x, start_y))
                if not os.path.exists(filepath_output):
                    create_folder(filepath_output)

                    df[c1 & c2 & c3].to_csv(filepath_output, index=False)
                    log("Save file in {}".format(filepath_output), INFO)
                else:
                    log("Skip {}".format(filepath_output), INFO)

def complex_split_data(filepath, time_column, time_func,
                       range_x=[x for x in range(0, 11, 1)], range_y=[y for y in range(0, 11, 1)],
                       size_x=[s for s in range(1, 6)], size_y=[y for y in range(1, 6)],
                       output_folder="."):

    df = pd.read_csv(filepath)
    df[time_column] = df["time"].map(time_func)

    Parallel(n_jobs=4)(delayed(_complex_split_data)(df, time_column, time_id, range_x, range_y, size_x, size_y, output_folder) for time_id in df[time_column].unique())

def _pos_split_data(df, x, range_y, window_size_x, window_size_y, output_folder):
    start_x, end_x = x, min(x+window_size_x, 11)
    c2 = (df["x"].values >= start_x) & (df["x"].values < end_x)

    for y in range_y:
         start_y, end_y = y, min(y+window_size_y, 11)
         c3 = (df["y"].values >= start_y) & (df["y"].values < end_y)

         filepath_output = os.path.join(output_folder, "windown_size={},{}".format(window_size_x, window_size_y), "{}_{}.csv".format(start_x, start_y))
         if not os.path.exists(filepath_output):
            create_folder(filepath_output)

            final_df = df[c2 & c3]
            if final_df.shape[0] > 0:
                final_df.to_csv(filepath_output, index=False)
                log("Save file in {}".format(filepath_output), INFO)

def pos_split_data(filepath, range_x=[x for x in range(0, 11, 1)], range_y=[y for y in range(0, 11, 1)],
                             size_x=[s for s in range(1, 6)], size_y=[y for y in range(1, 6)],
                             output_folder="."):

    df = pd.read_csv(filepath)
    for window_size_x, window_size_y in zip(size_x, size_y):
        Parallel(n_jobs=8)(delayed(_pos_split_data)(df, x, range_y, window_size_x, window_size_y, output_folder) for x in range_x)

def _place_info(df_all, place_id, output_folder):
    filepath_output = os.path.join(output_folder, "{}.csv".format(place_id))

    if os.path.exists(filepath_output):
        log("Skip {}".format(filepath_output), INFO)
    else:
        df = df_all[df_all["place_id"] == place_id]

        results = {"place_id": [place_id]}
        results["left_top"] = ["{},{}".format(df["x"].min(), df["y"].max())]
        results["right_bottom"] = ["{},{}".format(df["x"].max(), df["y"].min())]
        results["std_x"] = [df["x"].std()]
        results["std_y"] = [df["y"].std()]
        results["count"] = [df.shape[0]]
        results["centroid"] = ["{},{}".format(df["x"].median(), df["y"].median())]

        filepath_output = os.path.join(output_folder, "{}.csv".format(place_id))
        create_folder(filepath_output)

        pd.DataFrame.from_dict(results, orient="index").T.to_csv(filepath_output, index=False)
        log("Save file in {}".format(filepath_output), INFO)

def place_info(filepath, column_name, column_func, output_folder="."):
    df = pd.read_csv(filepath)

    df[column_name] = df["time"].map(column_func)
    for hour in df[column_name].unique():
        target_df = df[df[column_name] == hour]

        folder = os.path.join(output_folder, "{}={}".format(column_name, hour))
        Parallel(n_jobs=12)(delayed(_place_info)(target_df, place_id, folder) for place_id in target_df["place_id"].unique())

def _place_info_index(df, range_x, range_y, size_x, size_y, output_folder):
    for window_size_x, window_size_y in zip(size_x, size_y):
        folder = output_folder + "_windown_size={},{}".format(window_size_x, window_size_y)

        for x in range_x:
            start_x, end_x = x, min(11, x+window_size_x)
            c2 = (df["x"].values >= start_x) & (df["x"].values < end_x)

            for y in range_y:
                start_y, end_y = y, min(11, y+window_size_y)
                c3 = (df["y"].values >= start_y) & (df["y"].values < end_y)

                if df[c2 & c3].shape[0] > 0:
                    filepath_output = os.path.join(folder, "{}_{}.csv".format(start_x, start_y))
                    if not os.path.exists(filepath_output):
                        create_folder(filepath_output)

                        with open(filepath_output, "wb") as OUTPUT:
                            place_ids, counts = np.unique(df[c2 & c3]["place_id"].values, return_counts=True)
                            for place_id, count in zip(place_ids, counts):
                                OUTPUT.write("{},{}\n".format(place_id, count))

                        log("Save file in {}".format(filepath_output), INFO)
                    else:
                        log("Skip {}".format(filepath_output), INFO)

def place_info_index(filepath, column_name, column_func,
                     range_x=[x for x in range(0, 11, 1)], range_y=[y for y in range(0, 11, 1)],
                     size_x=[s for s in range(1, 6)], size_y=[y for y in range(1, 6)],
                     output_folder="."):
    df = pd.read_csv(filepath)
    df[column_name] = df["time"].map(column_func)

    Parallel(n_jobs=8)(delayed(_place_info_index)(df[df[column_name] == time_id], range_x, range_y, size_x, size_y, os.path.join(output_folder, "{}={}".format(column_name, time_id))) for time_id in df[time_column].unique())

def _time_split_data(df, new_column, value, output_folder):
    idx = (df[new_column] == value)

    filepath_output = os.path.join(output_folder, "{}.csv".format(value))
    create_folder(filepath_output)

    df[idx].to_csv(filepath_output, index=False)
    log("Save file in {}".format(filepath_output), INFO)

def time_split_data(filepath, column, column_func, new_column, output_folder):
    df = pd.read_csv(filepath)

    df[new_column] = df[column].map(column_func)

    Parallel(n_jobs=8)(delayed(_time_split_data)(df, new_column, value, output_folder) for value in df[new_column].unique())

def plot_place_history(filepath):
    history = {}

    with open(filepath, "rb") as INPUT:
        for line in INPUT:
            row_id, x, y, accuracy, time, place_id = line.strip().split(",")
            if not row_id.isdigit():
                continue

            key, value = (x, y), "{}({})".format(place_id, int(accuracy))
            history.setdefault(key, [])
            history[key].append(value)

    filepath_output = filepath.replace(".csv", ".history.csv")
    with open(filepath_output, "wb") as OUTPUT:
        for key, values in sorted(history.items(), key=lambda (k, v): len(v), reverse=True):
            OUTPUT.write("{},{}\n".format(key, "-".join(values)))

if __name__ == "__main__":
    mode = "original"

    folder = "/Users/rongqichen/Documents/programs/kaggle/cases/Facebook V - Predicting Check Ins/input/{}".format(mode)
    parent_folder = os.path.dirname(folder)

    filepath_train = "{}/train.csv".format(folder)
    filepath_test = "{}/test.csv".format(folder)

    '''
    range_x, size_x = [float(x)/10 for x in range(0, 110, 1)], [0.4]
    range_y, size_y = [float(x)/20 for x in range(0, 220, 1)], [0.1]
    '''

    '''
    range_x, size_x = [float(x)/20 for x in range(0, 220, 1)], [0.05]
    range_y, size_y = [float(x)/20 for x in range(0, 220, 1)], [0.05]
    '''

    range_x, size_x = [float(x)/20 for x in range(0, 220, 1)], [0.1]
    range_y, size_y = [float(x)/20 for x in range(0, 220, 1)], [0.1]

    time_column = "hourofday"
    time_func = lambda t: (t/60)%24

    place_info_index(filepath_train, time_column, time_func, range_x, range_y, size_x, size_y, output_folder=os.path.join(parent_folder, "place_index", "train"))
    place_info(filepath_train, time_column, time_func, output_folder=os.path.join(parent_folder, "place", "train"))

    #complex_split_data(filepath_train, time_column, time_func, range_x, range_y, size_x, size_y, output_folder=os.path.join(parent_folder, "2_way", "train"))
    #complex_split_data(filepath_test, time_column, time_func, range_x, range_y, size_x, size_y, output_folder=os.path.join(parent_folder, "2_way", "test"))

    #pos_split_data(filepath_train, range_x, range_y, size_x, size_y, output_folder=os.path.join(parent_folder, "train", "pos"))
    #pos_split_data(filepath_test, range_x, range_y, size_x, size_y, output_folder=os.path.join(parent_folder, "test", "pos"))

    #time_split_data(filepath_train, "time", time_func, time_column, os.path.join(parent_folder, "train", time_column))
    #time_split_data(filepath_test, "time", time_func, time_column, os.path.join(parent_folder, "test", time_column))

#!/usr/bin/python

import os
import glob
import time
import json
import subprocess

import numpy as np
import pandas as pd

from sklearn import linear_model

from utils import log, create_folder
from utils import INFO, ERROR
from bimbo.constants import COLUMN_AGENCY, COLUMN_CHANNEL, COLUMN_ROUTE, COLUMN_PRODUCT, COLUMN_CLIENT, COLUMN_PREDICTION, COLUMN_WEEK, COLUMN_ROW, MONGODB_COLUMNS, COLUMNS
from bimbo.constants import PYPY, SPLIT_PATH, FTLR_PATH, MEDIAN_SOLUTION_PATH, FTLR_SOLUTION_PATH, REGRESSION_SOLUTION_PATH, ROUTE_GROUPS, AGENCY_GROUPS, BATCH_JOB
from bimbo.constants import get_mongo_connection, get_median
from bimbo.model import Learning, LearningCost

def cache_median(filepath, filetype, week=9, output_folder=MEDIAN_SOLUTION_PATH):
    df = pd.read_csv(filepath)

    shape = df.shape
    df = df[df[COLUMN_WEEK] <= week]
    new_shape = df.shape
    log("After filtering, the shape is modified from {} to {}".format(shape, new_shape), INFO)

    drop_columns = [COLUMN_WEEK, 'Venta_uni_hoy', 'Venta_hoy', 'Dev_uni_proxima', 'Dev_proxima']
    df.drop(drop_columns, inplace=True, axis=1)

    target = {COLUMN_PREDICTION: np.median}

    groups = None
    if filetype == MONGODB_COLUMNS[COLUMN_ROUTE]:
        groups = ROUTE_GROUPS
    elif filetype == MONGODB_COLUMNS[COLUMN_AGENCY]:
        groups = AGENCY_GROUPS

    for group in groups:
        median = df.groupby(group).agg(target).to_dict()

        solution = {}
        for key, value in median[COLUMN_PREDICTION].items():
            if isinstance(key, np.int64):
                solution[str(key)] = value
            else:
                solution["_".join([str(s) for s in key])] = value

        log("There are {} records in median_solution".format(len(solution)), INFO)
        output_filepath = os.path.join(output_folder, filetype, "week={}".format(week), "{}.json".format("_".join([str(s) for s in group])))
        create_folder(output_filepath)
        with open(output_filepath, "wb") as OUTPUT:
            json.dump(solution, OUTPUT)

            log("Write median solution to {}".format(output_filepath), INFO)

def median_solution(week, output_filepath, filepath, solution):
    log("Store the solution in {}".format(output_filepath), INFO)
    create_folder(output_filepath)

    ts = time.time()
    with open(output_filepath, "wb") as OUTPUT:
        log("Read {}".format(filepath), INFO)
        header = True

        if week < 10:
            OUTPUT.write("Semana,Agencia_ID,Canal_ID,Ruta_SAK,Cliente_ID,Producto_ID,MEDIAN_Demanda_uni_equil\n")

            with open(filepath) as INPUT:
                for line in INPUT:
                    if header:
                        header = False
                    else:
                        w, agency_id, channel_id, route_id, client_id, product_id, _, _, _, _, _ = line.strip().split(",")

                        w = int(w)
                        if w == week:
                            prediction_median = get_median(solution[0], solution[1], {COLUMN_AGENCY: agency_id, COLUMN_PRODUCT: product_id, COLUMN_CLIENT: client_id})

                            OUTPUT.write("{}\n".format(",".join([str(w), agency_id, channel_id, route_id, client_id, product_id, str(prediction_median)])))
                        else:
                            pass
        else:
            OUTPUT.write("id,Demanda_uni_equil\n")

            with open(filepath, "rb") as INPUT:
                for line in INPUT:
                    if header:
                        header = False
                    else:
                        row_id, w, agency_id, channel_id, route_id, client_id, product_id = line.strip().split(",")
                        prediction_median = get_median(solution[0], solution[1], {COLUMN_AGENCY: agency_id, COLUMN_PRODUCT: product_id, COLUMN_CLIENT: client_id})

                        OUTPUT.write("{},{}\n".format(row_id, prediction_median))

    te = time.time()
    log("Cost {:4f} secends to generate the solution".format(te-ts), INFO)

def ftlr_solution(folder, fileid, submission_folder):
    cmd = "{} {} \"{}\" {} \"{}\"".format(PYPY, FTLR_PATH, folder, fileid, submission_folder)

    log("Start to predict {}/{}, and then exiting code is {}".format(\
        folder, fileid, subprocess.call(cmd, shell=True)), INFO)

def regression_solution(filetype, week_x, week_y, solutions=["ftlr", "median", "cc"], nfold=3):
    filepath_output = None
    if week_x != week_y:
        filepath_output = os.path.join(REGRESSION_SOLUTION_PATH, "test", "week={}".format(week_y), COLUMNS[filetype[0]], "submission_{}.csv".format(filetype[1]))
        create_folder(filepath_output)

        if os.path.exists(filepath_output):
            return
        else:
            log(filetype)

    subfolder = "train"
    index = ["Semana","Agencia_ID","Canal_ID","Ruta_SAK","Cliente_ID","Producto_ID"]

    solution_columns = []

    g = np.vectorize(lambda x: np.log1p(x))
    rg = np.vectorize(lambda x: np.e**max(1, x)-1)

    dfs = []
    for solution in solutions + ["real"]:
        filepath = None
        df = np.array([])

        if solution == "ftlr":
            filepath = os.path.join(FTLR_SOLUTION_PATH, subfolder, "week={}".format(week_x), COLUMNS[filetype[0]], "submission_{}.csv".format(filetype[1]))
            if os.path.exists(filepath) and os.path.getsize(filepath) > 84:
                log(filepath)
                df = pd.read_csv(filepath, index_col=index)

                df["FTLR_Demanda_uni_equil"] = g(df["FTLR_Demanda_uni_equil"])

                solution_columns.append("FTLR_Demanda_uni_equil")
        elif solution == "median":
            filepath = os.path.join(MEDIAN_SOLUTION_PATH, subfolder, "week={}".format(week_x), COLUMNS[filetype[0]], "submission_{}.csv".format(filetype[1]))
            if os.path.exists(filepath) and os.path.getsize(filepath) > 84:
                df = pd.read_csv(filepath, index_col=index)

                df["MEDIAN_Demanda_uni_equil"] = g(df["MEDIAN_Demanda_uni_equil"])

                solution_columns.append("MEDIAN_Demanda_uni_equil")
        elif solution == "cc":
            filepath = os.path.join(MEDIAN_SOLUTION_PATH, subfolder, "week={}".format(week_x), COLUMNS[filetype[0]], "submission_{}.csv".format(filetype[1]))
            if os.path.exists(filepath) and os.path.getsize(filepath) > 84:
                df = pd.read_csv(filepath, index_col=index)

                #g_max = np.vectorize(lambda x: max(x, 1))
                #df["CC_Demanda_uni_equil"] = g_max(df["CC_Demanda_uni_equil"].values)

                df["CC_Demanda_uni_equil"] = g(df["CC_Demanda_uni_equil"])

                solution_columns.append("CC_Demanda_uni_equil")
        elif solution == "real":
            filepath = os.path.join(SPLIT_PATH, COLUMNS[filetype[0]], "train", "{}.csv".format(filetype[1]))
            if os.path.exists(filepath) and os.path.getsize(filepath) > 84:
                log(filepath)
                df = pd.read_csv(filepath)
                df.drop(["Venta_uni_hoy","Venta_hoy","Dev_uni_proxima","Dev_proxima"], axis=1, inplace=True)

                df = df[df[COLUMN_WEEK] == week_x]
                if df.shape[0] > 0:
                    df["Demanda_uni_equil"] = g(df["Demanda_uni_equil"])

                df = df.set_index(index)
        else:
            raise NotImplementedError

        if df.shape[0] > 0:
            dfs.append(df)
            log("Read {} completely for {}".format(filepath, solution), INFO)

    training = np.array([])
    if dfs:
        training = dfs[0]
        for result in dfs[1:]:
            training = training.join(result)

    predicted_x = np.array([])
    if week_x == week_y:
        predicted_x = training[solution_columns].values
    else:
        subfolder = "test"
        index = ["id"]

        dfs = []
        for solution in solutions:
            if solution == "ftlr":
                filepath = os.path.join(FTLR_SOLUTION_PATH, subfolder, "week={}".format(week_y), COLUMNS[filetype[0]], "submission_{}.csv".format(filetype[1]))
                df = pd.read_csv(filepath, index_col=index)
                df["Demanda_uni_equil"] = g(df["Demanda_uni_equil"])
                df.rename(columns={"Demanda_uni_equil": "FTLR_Demanda_uni_equil"}, inplace=True)
            elif solution == "median":
                filepath = os.path.join(MEDIAN_SOLUTION_PATH, subfolder, "week={}".format(week_y), COLUMNS[filetype[0]], "submission_{}.csv".format(filetype[1]))
                df = pd.read_csv(filepath, index_col=index)
                df["Demanda_uni_equil"] = g(df["Demanda_uni_equil"])
                df.rename(columns={"Demanda_uni_equil": "MEDIAN_Demanda_uni_equil"}, inplace=True)
            elif solution == "cc":
                filepath = os.path.join(MEDIAN_SOLUTION_PATH, subfolder, "week={}".format(week_y), COLUMNS[filetype[0]], "submission_{}.csv".format(filetype[1]))
                df = pd.read_csv(filepath, index_col=index)
                df["Demanda_uni_equil"] = g(df["Demanda_uni_equil"])
                df.rename(columns={"Demanda_uni_equil": "CC_Demanda_uni_equil"}, inplace=True)
            else:
                log("The solution is {}".format(solution), ERROR)
                raise NotImplementedError

            dfs.append(df)

        testing = dfs[0]
        for result in dfs[1:]:
            testing = testing.join(result)

        predicted_x = testing[solution_columns].values

    if training.shape[0] >= nfold:
        dataset_x = np.array(training[solution_columns].values)
        dataset_y = np.array(training["Demanda_uni_equil"].values)
        models = [linear_model.LinearRegression(), linear_model.Ridge(alpha=1.0)]

        learning = Learning(dataset_x, dataset_y, predicted_x, models, nfold)

        for clf in learning.get_models():
            log("The coef of {} is {}".format(clf, clf.coef_), INFO)

        single_rmsle = []
        for solution in solution_columns:
            single_rmsle.append(str(LearningCost.rmsle_2(dataset_y, training[solution].values)))

        log("The RMSLE({}) is {}/{}".format(filetype, "/".join(single_rmsle), LearningCost.rmsle_2(dataset_y, learning.get_results(False))), INFO)

        if filepath_output:
            testing = testing.reset_index(level=["id"])

            log("Start to output the prediction results to {}".format(filepath_output), INFO)
            with open(filepath_output, "wb") as OUTPUT:
                OUTPUT.write("id,Demanda_uni_equil\n")

                for row_id, value in zip(testing["id"], rg(learning.predict())):
                    OUTPUT.write("{},{}\n".format(row_id, value))
        else:
            log("Skip the step of prediction results output", INFO)
    else:
        if filepath_output:
            testing = testing.reset_index(level=["id"])

            log("Start to output the prediction results to {}".format(filepath_output), INFO)
            with open(filepath_output, "wb") as OUTPUT:
                OUTPUT.write("id,Demanda_uni_equil\n")

                for row_id, value in zip(testing["id"], rg(np.mean(testing[solution_columns].values, axis=1))):
                    OUTPUT.write("{},{}\n".format(row_id, value))
        else:
            log("Skip the step of prediction results output", INFO)

def ensemble_solution(filepaths, output_filepath):
    frames = []
    for filepath in filepaths:
        log("Start to read {}".format(filepath), INFO)
        df = pd.read_csv(filepath)

        frames.append(df)

    # Header
    # id,Demanda_uni_equil

    result = pd.concat(frames)
    target = {COLUMN_PREDICTION: np.mean}

    result.groupby(["id"]).agg(target).to_csv(output_filepath)

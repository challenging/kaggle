#!/usr/bin/python

import os
import glob
import gzip
import time
import json
import subprocess

import numpy as np
import pandas as pd

from utils import log, create_folder
from utils import INFO
from bimbo.constants import COLUMN_AGENCY, COLUMN_CHANNEL, COLUMN_ROUTE, COLUMN_PRODUCT, COLUMN_CLIENT, COLUMN_PREDICTION, COLUMN_WEEK, COLUMN_ROW, MONGODB_COLUMNS, COLUMNS
from bimbo.constants import PYPY, MEDIAN_SOLUTION_PATH, FTLR_SOLUTION_PATH, ROUTE_GROUPS, AGENCY_GROUPS, BATCH_JOB
from bimbo.constants import get_mongo_connection, get_median

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

def median_solution(output_filepath, filepath_test, solution):
    log("Store the solution in {}".format(output_filepath), INFO)
    create_folder(output_filepath)

    ts = time.time()
    with gzip.open(output_filepath, "wb") as OUTPUT:
        OUTPUT.write("id,Demanda_uni_equil\n")

        for filepath in glob.iglob(filepath_test):
            log("Read {}".format(filepath), INFO)
            header = True

            with open(filepath, "rb") as INPUT:
                for line in INPUT:
                    if header:
                        header = False
                    else:
                        row_id, w, agency_id, channel_id, route_id, client_id, product_id = line.strip().split(",")

                        prediction_median = get_median(solution[0], solution[1], {COLUMN_ROUTE: route_id, COLUMN_PRODUCT: product_id, COLUMN_CLIENT: client_id})

                        OUTPUT.write("{},{}\n".format(row_id, prediction_median))

    te = time.time()
    log("Cost {:4f} secends to generate the solution".format(te-ts), INFO)

def ftlr_solution(folder, fileid, submission_folder):
    cmd = "{} {} \"{}\" {} \"{}\"".format(PYPY, "../lib/bimbo/ftlr.py", folder, fileid, submission_folder)

    log("Start to predict {}/{}, and then exiting code is {}".format(\
        folder, fileid, subprocess.call(cmd, shell=True)), INFO)

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

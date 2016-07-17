#!/usr/bin/python

import os
import sys
import json
import subprocess

import numpy as np
import pandas as pd

from utils import log
from utils import INFO
from bimbo.constants import COLUMN_AGENCY, COLUMN_CHANNEL, COLUMN_ROUTE, COLUMN_PRODUCT, COLUMN_CLIENT, COLUMN_PREDICTION, COLUMN_WEEK, COLUMN_ROW, MONGODB_COLUMNS, COLUMNS
from bimbo.constants import MEDIAN_SOLUTION_PATH, FTLR_SOLUTION_PATH, ROUTE_GROUPS

def median_solution(filepath, week=9, output_folder=MEDIAN_SOLUTION_PATH):
    df = pd.read_csv(filepath)

    shape = df.shape
    df = df[df[COLUMN_WEEK] <= week]
    new_shape = df.shape
    log("After filtering, the shape is modified from {} to {}".format(shape, new_shape), INFO)

    drop_columns = [COLUMN_WEEK, 'Venta_uni_hoy', 'Venta_hoy', 'Dev_uni_proxima', 'Dev_proxima']
    df.drop(drop_columns, inplace=True, axis=1)

    target = {COLUMN_PREDICTION: np.median}

    for group in ROUTE_GROUPS:
        median = df.groupby(group).agg(target).to_dict()

        solution = {}
        for key, value in median[COLUMN_PREDICTION].items():
            if isinstance(key, np.int64):
                solution[str(key)] = value
            else:
                solution["_".join([str(s) for s in key])] = value

        log("There are {} records in median_solution".format(len(solution)), INFO)
        output_filepath = os.path.join(output_folder, "week={}".format(week), "{}.json".format("_".join([str(s) for s in group])))
        with open(output_filepath, "wb") as OUTPUT:
            json.dump(solution, OUTPUT)

            log("Write median solution to {}".format(output_filepath), INFO)

def ftlr_solution(folder, fileid, submission_folder):
    cmd = "{} {} \"{}\" {} \"{}\"".format(PYPY, "ftlr.py", folder, fileid, submission_folder)

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

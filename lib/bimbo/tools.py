#!/usr/bin/env python

import os
import sys
import json
import glob
import shutil
import socket
import time
import click
import pymongo
import subprocess

# for debug
import pprint

import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from utils import log, create_folder
from utils import DEBUG, INFO, WARN
from bimbo.cc_beanstalk import cc_calculation
from bimbo.constants import get_stats_mongo_collection, get_mongo_connection
from bimbo.constants import COLUMN_AGENCY, COLUMN_CHANNEL, COLUMN_ROUTE, COLUMN_PRODUCT, COLUMN_CLIENT, COLUMN_PREDICTION, COLUMN_WEEK, COLUMN_ROW, MONGODB_COLUMNS, COLUMNS
from bimbo.constants import PYPY, IP_BEANSTALK, MONGODB_DATABASE, MONGODB_BATCH_SIZE
from bimbo.constants import MEDIAN_SOLUTION_PATH, FTLR_SOLUTION_PATH, SPLIT_PATH, STATS_PATH, TRAIN_FILE, TEST_FILE, TESTING_TRAIN_FILE, TESTING_TEST_FILE
from bimbo.constants import NON_PREDICTABLE

TRAIN = TRAIN_FILE
TEST = TEST_FILE

def purge_duplicated_records(column, batch_size=MONGODB_BATCH_SIZE):
    client = get_mongo_connection()
    collection = client[MONGODB_DATABASE][get_stats_mongo_collection(column)]

    count = 0
    pre_row_id, pre_object_id = None, None
    for record in collection.find({}, {"_id": 1, COLUMN_ROW: 1}).sort([(COLUMN_ROW, pymongo.ASCENDING)]).batch_size(batch_size):
        row_id, object_id = record[COLUMN_ROW], record["_id"]

        if pre_row_id == row_id:
            count += 1

            log("Delete {}({} - {}), accumulated size is {}".format(collection.remove({"_id": pre_object_id}), pre_row_id, pre_object_id, count), INFO)

        pre_row_id, pre_object_id = row_id, object_id

    log("Delete {} records".format(count), INFO)

    client.close()

def repair_missing_records(column, batch_size=MONGODB_BATCH_SIZE):
    client = get_mongo_connection()
    collection = client[MONGODB_DATABASE][get_stats_mongo_collection(column)]

    row_ids = set()
    for record in collection.find({}, {"_id": 1, COLUMN_ROW: 1}).sort([(COLUMN_ROW, pymongo.ASCENDING)]).batch_size(batch_size):
        row_id = record[COLUMN_ROW]
        row_ids.add(row_id)

    records = []
    with open(TEST_FILE, "rb") as INPUT:
        header = True

        #id,Semana,Agencia_ID,Canal_ID,Ruta_SAK,Cliente_ID,Producto_ID
        for line in INPUT:
            if header:
                header = False
                continue

            row_id, week_num, agency_id, channel_id, route_id, client_id, product_id = line.strip().split(",")
            row_id = int(row_id)
            week_num = int(week_num)
            agency_id = int(agency_id)
            channel_id = int(channel_id)
            route_id = int(route_id)
            client_id = int(client_id)
            product_id = int(product_id)

            if row_id not in row_ids:
                record = {
                    "row_id": row_id,
                    "fixed_column": column,
                    "week_num": week_num,
                    "matching_count": 0,
                    MONGODB_COLUMNS[COLUMN_AGENCY]: agency_id,
                    MONGODB_COLUMNS[COLUMN_CHANNEL]: channel_id,
                    MONGODB_COLUMNS[COLUMN_ROUTE]: route_id,
                    MONGODB_COLUMNS[COLUMN_CLIENT]: client_id,
                    MONGODB_COLUMNS[COLUMN_PRODUCT]: product_id,
                 }

                log(record, INFO)
                records.append(record)

    collection.insert_many(records)
    log("Add {} records into the {}".format(len(records), get_stats_mongo_collection(column)), INFO)

    client.close()

def hierarchical_folder_structure(column, filetype):
    prefixs = set()
    folder = os.path.join(SPLIT_PATH, COLUMNS[column], filetype.lower())

    if not os.path.isdir(folder):
        log("{} is not a folder".format(folder), INFO)
        return

    timestamp_start = time.time()
    for filepath in glob.iglob("{}/*.csv".format(folder)):
        filename = os.path.basename(filepath)

        prefix = filename[0:3]
        prefixs.add(prefix)

        new_folder = os.path.join(folder, prefix)
        new_filepath = os.path.join(new_folder, filename)

        create_folder(new_filepath)
        os.rename(filepath, new_filepath)
        log("Move {} to {}".format(filepath, new_filepath), INFO)

    timestamp_end = time.time()
    log("Cost {:4f} secends to move files to the sub-folders".format(timestamp_end-timestamp_start), INFO)

    hostname = socket.gethostname()
    if hostname != IP_BEANSTALK:
        timestamp_start = time.time()
        for prefix in prefixs:
            filepath = os.path.join(folder, prefix)

            p = subprocess.Popen(["scp", "-r", filepath, "RungChiChen@{}:\"{}\"".format(IP_BEANSTALK, folder.replace(" ", "\\\\ "))])
            pid, sts = os.waitpid(p.pid, 0)
            log("Transfer {} successfully({})".format(filepath, sts), INFO)

    timestamp_end = time.time()
    log("Cost {:4f} secends to copy files to the {}".format(timestamp_end-timestamp_start, IP_BEANSTALK), INFO)

def aggregation(group_columns, output_filepath):
    global TRAIN

    df_train = pd.read_csv(TRAIN)

    target = {"Venta_uni_hoy":sum, "Dev_uni_proxima":sum, "Demanda_uni_equil": sum}

    log("The group columns are {}".format(group_columns), INFO)
    log("The target is {}".format(target), INFO)

    df = df_train.groupby(group_columns).agg(target)
    df.to_csv(output_filepath)

def cc(filepath, filetype, threshold_value=0):
    shift_week = 3
    history = {}

    def all_zero_list():
        return [0 for _ in range(3, 10)]

    with open(filepath, "rb") as INPUT:
        header = True

        for line in INPUT:
            if header:
                header = False
                continue

            week, agency_id, channel_id, route_id, client_id, product_id, sales_unit, sales_price, return_unit, return_price, prediction_unit = line.strip().split(",")

            week = int(week)
            client_id = int(client_id)
            product_id = int(product_id)
            prediction_unit = int(prediction_unit)

            key = product_id

            history.setdefault(key, {})
            history[key].setdefault(client_id, all_zero_list())
            history[key][client_id][week-shift_week] = prediction_unit
            #history[key][client_id][week-shift_week] = np.log1p(prediction_unit)

    loss_sum, loss_count = 0, 0
    for no, (key, info) in enumerate(history.items()):
        for rtype, record, lsum in cc_calculation(key, info, threshold_value, progress_prefix=(no+1, len(history)), alternative_filetype=filetype[0], alternative_id=filetype[1]):
            if rtype in ["cc", "median"]:
                loss_sum += lsum
                loss_count += 1

            elif rtype == NON_PREDICTABLE:
                pass

    log("Total RMLSE: {:8f}".format(loss_sum/loss_count), INFO)

def median_solution(folder, output_filepaths):
    global_route_prod_client_solution = {}
    global_route_prod_solution = {}

    for filepath in glob.iglob(os.path.join(folder, "*.csv")):
        log("Start to process {}".format(filepath))

        df = pd.read_csv(filepath)

        drop_columns = [COLUMN_WEEK, 'Venta_uni_hoy', 'Venta_hoy', 'Dev_uni_proxima', 'Dev_proxima']
        df.drop(drop_columns, inplace=True, axis=1)

        target = {COLUMN_PREDICTION: np.median}

        route_prod_client_median = df.groupby([COLUMN_ROUTE, COLUMN_PRODUCT, COLUMN_CLIENT]).agg(target).to_dict()
        for key, value in route_prod_client_median[COLUMN_PREDICTION].items():
            global_route_prod_client_solution["_".join([str(s) for s in key])] = value

        route_prod_median = df.groupby([COLUMN_ROUTE, COLUMN_PRODUCT]).agg(target).to_dict()
        for key, value in route_prod_median[COLUMN_PREDICTION].items():
            global_route_prod_solution["_".join([str(s) for s in key])] = value

    log("Have {}/{} records in global_naive_solution".format(len(global_route_prod_client_solution), len(global_route_prod_solution)), INFO)
    for solution, filepath in zip([global_route_prod_client_solution, global_route_prod_solution], output_filepaths):
        create_folder(filepath)

        with open(filepath, "wb") as OUTPUT:
            json.dump(solution, OUTPUT)

            log("Write naive-global solution to {}".format(filepath), INFO)

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

@click.command()
@click.option("--is-testing", is_flag=True, help="testing mode")
@click.option("--column", default=None, help="agency_id|channel_id|route_id|client_id|product_id")
@click.option("--mode", required=True, help="purge|restructure")
@click.option("--option", required=False, nargs=2, type=click.Tuple([unicode, unicode]), default=(None, None))
def tool(is_testing, column, mode, option):
    global TRAIN, TEST

    if is_testing:
        TRAIN = TESTING_TRAIN_FILE
        TEST = TESTING_TEST_FILE

    if mode == "purge":
        purge_duplicated_records(column)
    elif mode == "restructure":
        for filetype in ["train", "test"]:
            hierarchical_folder_structure(column, filetype)
    elif mode == "repair":
        repair_missing_records(column)
    elif mode == "aggregation":
        columns = [COLUMNS[c] for c in column.split(",")]
        output_filepath = os.path.join(STATS_PATH, "{}.csv".format("_".join(columns)))
        create_folder(output_filepath)

        aggregation(columns, output_filepath)
    elif mode == "cc":
        column, column_value = option
        column_value = int(column_value)

        filepath = os.path.join(SPLIT_PATH, COLUMNS[column], "train", "{}.csv".format(column_value))

        cc(filepath, ("{}_product".format(column), column_value))
    elif mode == "median":
        folder = os.path.join(SPLIT_PATH, COLUMNS[column], "train")
        output_filepaths = [os.path.join(MEDIAN_SOLUTION_PATH, "{}_product_client.json".format(column)),
                            os.path.join(MEDIAN_SOLUTION_PATH, "{}_product.json".format(column))]

        median_solution(folder, output_filepaths)
    elif mode == "ftlr":
        folder = os.path.join(SPLIT_PATH, COLUMNS[column], "test")
        submission_folder = os.path.join(FTLR_SOLUTION_PATH, COLUMNS[column])
        create_folder("{}/1.txt".format(submission_folder))

        Parallel(n_jobs=6)(delayed(ftlr_solution)(folder, os.path.basename(filepath).replace(".csv", ""), submission_folder) for filepath in glob.iglob(os.path.join(folder, "*.csv")))
    elif mode == "ensemble":
        filepaths, output_filepath = option

        ensemble_solution(filepaths.split(","), output_filepath)
    else:
        log("Not found this mode {}".format(mode), ERROR)
        sys.exit(101)

if __name__ ==  "__main__":
    tool()

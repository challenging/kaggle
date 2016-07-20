#!/usr/bin/env python

import os
import glob
import socket
import time
import pymongo
import subprocess

import numpy as np
import pandas as pd

from utils import log, create_folder
from utils import DEBUG, INFO, WARN
from bimbo.cc_beanstalk import cc_calculation, get_history, get_median
from bimbo.constants import get_stats_mongo_collection, get_mongo_connection, load_median_route_solution
from bimbo.constants import COLUMN_AGENCY, COLUMN_CHANNEL, COLUMN_ROUTE, COLUMN_PRODUCT, COLUMN_CLIENT, COLUMN_PREDICTION, COLUMN_WEEK, COLUMN_ROW, MONGODB_COLUMNS, COLUMNS
from bimbo.constants import TOTAL_WEEK, PYPY, IP_BEANSTALK, MONGODB_DATABASE, MONGODB_BATCH_SIZE, SPLIT_PATH, NON_PREDICTABLE

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

def aggregation(filepath, group_columns, output_filepath):
    df_train = pd.read_csv(filepath)

    target = {"Venta_uni_hoy":sum, "Dev_uni_proxima":sum, "Demanda_uni_equil": sum}

    log("The group columns are {}".format(group_columns), INFO)
    log("The target is {}".format(target), INFO)

    df = df_train.groupby(group_columns).agg(target)
    df.to_csv(output_filepath)

def cc_solution(week, filepath_train, filepath_test, filetype, median_solution, threshold_value=0):
    global TOTAL_WEEK

    shift_week = 3
    end_idx = (TOTAL_WEEK-shift_week) - (TOTAL_WEEK-week)

    history, predicted_rows = get_history(filepath_train, filepath_test, shift_week=shift_week)

    sign_plus, sign_minus = 0, 0
    ts = time.time()
    rmsle_mean, rmsle_cc, rmsle_median, loss_count = 0, 0, 0, 0.00000001
    for no, (product_id, info) in enumerate(history.items()):
        partial_rmsle_mean, partial_rmsle_cc, partial_rmsle_median, partial_loss_count = 0, 0, 0, 0.00000001

        for record, prediction in cc_calculation(week, filetype, product_id, predicted_rows[product_id], info, threshold_value, (no+1, len(history))):
            client_id = record["client_id"]

            true_value = history[product_id][client_id][end_idx]
            if true_value == 0:
                continue

            prediction_median = get_median(median_solution[0], median_solution[1], {filetype[0]: filetype[1], COLUMN_PRODUCT: product_id, COLUMN_CLIENT: client_id})
            prediction_cc = prediction["prediction_cc"]

            if prediction_cc < 0:
                prediction_cc = 0
            else:
                prediction_cc = prediction_cc*0.71+1.5

            prediction_mean = (prediction_median+prediction_cc)/2

            loss_cc = (np.log1p(prediction_cc)-np.log1p(true_value))**2
            loss_median = (np.log1p(prediction_median)-np.log1p(true_value))**2
            loss_mean = (np.log1p(prediction_mean)-np.log1p(true_value))**2

            if prediction_cc > true_value:
                sign_plus += loss_cc
            elif prediction_cc < true_value:
                sign_minus += loss_cc

            log("Predict {} - {} - {}, {}/{}/{}".format(product_id, client_id, history[product_id][client_id], prediction_cc, prediction_median, prediction_mean), INFO)

            partial_rmsle_cc += loss_cc
            partial_rmsle_median += loss_median
            partial_rmsle_mean += loss_mean
            partial_loss_count += 1

        log("RMSLE of {} is {}/{}/{}".format(product_id, np.sqrt(partial_rmsle_cc/partial_loss_count), np.sqrt(partial_rmsle_median/partial_loss_count), np.sqrt(partial_rmsle_mean/partial_loss_count)), INFO)

        rmsle_cc += partial_rmsle_cc
        rmsle_median += partial_rmsle_median
        rmsle_mean += partial_rmsle_mean
        loss_count += partial_loss_count

    loss_count = int(loss_count)
    te = time.time()

    log("Cost {:4f} secends to get the total RMLSE: {:4f}/{:4f}/{:4f}".format(te-ts, np.sqrt(rmsle_cc/loss_count), np.sqrt(rmsle_median/loss_count), np.sqrt(rmsle_mean/loss_count)), INFO)
    log("{} vs. {}".format(sign_plus, sign_minus))

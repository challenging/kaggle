#!/usr/bin/env python

import os
import glob
import socket
import time
import pymongo
import subprocess

import numpy as np
import pandas as pd

from sklearn import linear_model

from utils import log, create_folder
from utils import DEBUG, INFO, WARN
from bimbo.cc_beanstalk import cc_calculation, get_history
from bimbo.constants import get_stats_mongo_collection, get_mongo_connection, get_prediction_mongo_collection
from bimbo.constants import COLUMN_AGENCY, COLUMN_CHANNEL, COLUMN_ROUTE, COLUMN_PRODUCT, COLUMN_CLIENT, COLUMN_PREDICTION, COLUMN_WEEK, COLUMN_ROW, MONGODB_COLUMNS, COLUMNS
from bimbo.constants import TOTAL_WEEK, PYPY, IP_BEANSTALK, MONGODB_DATABASE, MONGODB_PREDICTION_DATABASE, MONGODB_BATCH_SIZE, SPLIT_PATH, NON_PREDICTABLE

from bimbo.model import Learning, LearningCost

def purge_duplicated_records(week, solution_type, column, batch_size=MONGODB_BATCH_SIZE):
    client = get_mongo_connection()
    collection = client["{}_{}".format(MONGODB_DATABASE, week)][get_prediction_mongo_collection(column)]

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

def cc_solution(week, filepath_train, filepath_test, filetype, threshold_value=0):
    global TOTAL_WEEK

    shift_week = 3
    end_idx = (TOTAL_WEEK-shift_week) - (TOTAL_WEEK-week)

    history, predicted_rows, others = get_history(week, filepath_train, filepath_test, shift_week=shift_week)

    ts = time.time()
    rmsle_cc, loss_count = 0, 0.00000001
    for no, (product_id, info) in enumerate(history.items()):
        partial_rmsle_cc, partial_loss_count = 0, 0

        for client_id, prediction in cc_calculation(week, filetype, product_id, predicted_rows[product_id], info, threshold_value, (no+1, len(history))):
            true_value = history[product_id][client_id][end_idx]
            if true_value == 0 or len(history[product_id]) == 1 or np.sum(history[product_id][client_id][0:end_idx]) == 0:
                prediction_cc = 0
            else:
                prediction_cc = prediction["prediction_cc"]

                loss_cc = (prediction_cc-true_value)**2

                log("Predict {} - {} - {}, {}".format(product_id, client_id, history[product_id][client_id], np.e**prediction_cc-1), DEBUG)

                partial_rmsle_cc += loss_cc
                partial_loss_count += 1

            for row in others["{}_{}".format(client_id, product_id)]:
                if row.startswith("{},".format(week)):
                    yield "{},{}".format(row, np.e**prediction_cc-1)

        if partial_loss_count > 0:
            log("RMSLE({}) of {} is {}".format(partial_loss_count, product_id, np.sqrt(partial_rmsle_cc/partial_loss_count)), INFO)

        rmsle_cc += partial_rmsle_cc
        loss_count += partial_loss_count

    te = time.time()

    log("Cost {:4f} secends to get the total RMLSE: {:4f}".format(te-ts, np.sqrt(rmsle_cc/loss_count)), INFO)

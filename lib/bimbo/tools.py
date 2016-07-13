#!/usr/bin/env python

import os
import sys
import glob
import shutil
import socket
import time
import click
import pymongo
import subprocess

import numpy as np
import pandas as pd

from utils import log, create_folder
from utils import DEBUG, INFO, WARN
from bimbo.constants import get_stats_mongo_collection, get_mongo_connection
from bimbo.constants import COLUMN_AGENCY, COLUMN_CHANNEL, COLUMN_ROUTE, COLUMN_PRODUCT, COLUMN_CLIENT, COLUMN_ROW, MONGODB_COLUMNS, COLUMNS
from bimbo.constants import IP_BEANSTALK, MONGODB_DATABASE, MONGODB_BATCH_SIZE, SPLIT_PATH, STATS_PATH, TRAIN_FILE, TEST_FILE, TESTING_TRAIN_FILE, TESTING_TEST_FILE

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

def cc(filepath, threshold_value=0):
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

    cc_calculation(history, threshold_value)

def cc_calculation(history, threshold_value=0):
    loss_median_sum = 0
    loss_cc_sum, loss_count = 0, 0
    for count_key, (product_id, info) in enumerate(zip(history.keys()[::-1], history.values()[::-1])):
        for client_id, values in info.items():
            client_mean = np.mean(values[1:-1])

            cc_client_ids, cc_matrix = [client_id], [values[1:-1]]
            for cc_client_id, cc_client_values in info.items():
                if client_id != cc_client_id:
                    cc_client_ids.append(cc_client_id)
                    cc_matrix.append(cc_client_values[0:-2])

            prediction_median = np.median(history[product_id][client_id][:-1])
            prediction_cc = prediction_median
            if np.sum(np.array(history[product_id][client_id]) == 0) > 4:
                prediction_cc = 0
            elif len(cc_client_ids) > 1:
                cc_results = np.corrcoef(cc_matrix)[0]
                results_cc = dict(zip(cc_client_ids, cc_results))

                num_sum, num_count = 0, 0.00000001
                for c, value in sorted(results_cc.items(), key=lambda (k, v): abs(v), reverse=True):
                    if c == client_id or np.isnan(value):
                        continue
                    elif abs(value) > threshold_value :
                        ratio = client_mean/np.mean(history[product_id][c][0:-2])
                        score = (history[product_id][c][-2] - history[product_id][c][-3])*value*ratio
                        num_sum += score
                        num_count += 1

                        #log("a. {} - {} - {}({}) - {} - {} - {} - {} - {}".format(product_id, client_id, c, history[product_id][c], value, score, ratio, num_sum, num_count), INFO)
                    else:
                        break

                prediction_unit = history[product_id][client_id][-2] + num_sum/num_count
                prediction_cc = max(0, prediction_unit)
            else: # only one person
                pass

            loss_count += 1
            loss_cc_sum += (np.log(prediction_cc+1)-np.log(values[-1]+1))**2
            loss_median_sum += (np.log(prediction_median+1)-np.log(values[-1]+1))**2

            log("{}/{}/{} >>> {} - {} - {} - {:4f}({:4f}) - {:4f}({:4f})".format(\
                loss_count, count_key, len(history), product_id, client_id, history[product_id][client_id], prediction_cc, prediction_median, np.sqrt(loss_cc_sum/loss_count), np.sqrt(loss_median_sum/loss_count)), INFO)

    log("Total RMLSE: {:6f}".format(np.sqrt(loss_cc_sum/loss_count)), INFO)

@click.command()
@click.option("--is-testing", is_flag=True, help="testing mode")
@click.option("--column", default=None, help="agency_id|channel_id|route_id|client_id|product_id")
@click.option("--mode", required=True, help="purge|restructure")
@click.option("--option", nargs=2, type=click.Tuple([unicode, int]))
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
        filepath = os.path.join(SPLIT_PATH, COLUMNS[column], "train", "{}.csv".format(column_value))

        cc(filepath)
    else:
        log("Not found this mode {}".format(mode), ERROR)
        sys.exit(101)

if __name__ ==  "__main__":
    tool()

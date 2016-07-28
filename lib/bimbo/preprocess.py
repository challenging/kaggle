#!/usr/bin/env python

import os
import sys
import json
import time
import beanstalkc

import threading
import Queue

import numpy as np
import pandas as pd

from utils import create_folder, log, INFO, WARN
from bimbo.constants import load_median_solution, get_median
from bimbo.constants import ROUTE_GROUPS, AGENCY_GROUPS, COLUMNS, COLUMN_WEEK, COLUMN_AGENCY, COLUMN_CHANNEL, COLUMN_ROUTE, COLUMN_PRODUCT, COLUMN_CLIENT
from bimbo.constants import SPLIT_PATH, TRAIN_FILE, TEST_FILE, COMPETITION_GROUP_NAME, COMPETITION_FEATURE_ENGINEER_NAME
from bimbo.constants import IP_BEANSTALK, PORT_BEANSTALK, TIMEOUT_BEANSTALK

class SplitThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        threading.Thread.__init__(self, group=group, target=target, name=name, verbose=verbose)

        self.args = args

        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def matching(df, column, value, median_route_solution, median_agency_solution):
        df = df[df[column] == value]

        fr = lambda x: get_median(median_route_solution[0], median_route_solution[1], {COLUMN_ROUTE: x[0], COLUMN_PRODUCT: x[1], COLUMN_CLIENT: x[2]})
        fa = lambda x: get_median(median_agency_solution[0], median_agency_solution[1], {COLUMN_AGENCY: x[0], COLUMN_PRODUCT: x[1], COLUMN_CLIENT: x[2]})

        df["median_route_solution"] = np.apply_along_axis(fr, 1, df[[COLUMN_ROUTE, COLUMN_PRODUCT, COLUMN_CLIENT]])
        df["median_agency_solution"] = np.apply_along_axis(fa, 1, df[[COLUMN_AGENCY, COLUMN_PRODUCT, COLUMN_CLIENT]])

        return df

    def run(self):
        while True:
            timestamp_start = time.time()

            output_filepath, filetype, column, value = self.queue.get()

            if os.path.exists(output_filepath):
                log("Found {} so skipping it".format(output_filepath), INFO)
            else:
                df = None
                if hasattr(self, "df"):
                    df = self.df
                elif filetype:
                    if filetype == "train":
                        df = self.df_train
                    elif filetype == "test":
                        df = self.df_test
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError

                self.matching(df, column, value, self.median_route_solution, self.median_agency_solution).to_csv(output_filepath, index=False)

            self.queue.task_done()

            timestamp_end = time.time()
            log("Cost {:4f} secends to store {}={} in {}, the remaining queue size is {}".format(timestamp_end-timestamp_start, column, value, output_filepath, self.queue.qsize()), INFO)

def producer(columns, ip=IP_BEANSTALK, port=PORT_BEANSTALK, task=COMPETITION_GROUP_NAME):
    talk = beanstalkc.Connection(host=ip, port=port)
    talk.watch(task)

    for filetype, filepath in zip(["train", "test"], [TRAIN_FILE, TEST_FILE]):
        log("Start to read {}".format(filepath), INFO)
        df = pd.read_csv(filepath)

        for column in columns.split(","):
            column = COLUMNS[column]

            output_folder = os.path.join(SPLIT_PATH, column, os.path.basename(filepath).replace(".csv", ""))

            output_filepaths, values = [], []
            for unique_value in df[column].unique():
                output_filepath = os.path.join(output_folder, "{}.csv".format(unique_value))

                if os.path.exists(output_filepath):
                    log("Found {} so skipping it".format(output_filepath), INFO)
                else:
                    output_filepaths.append(output_filepath)
                    values.append(unique_value)

                    if len(values) > 2**7-1:
                        request = {"filetype": filetype,
                                   "output_filepath": output_filepaths,
                                   "column": column,
                                   "value": values}
                        talk.put(json.dumps(request), ttr=3600)
                        log("Put {} requests into the queue".format(len(values)), INFO)

                        output_filepaths, values = [], []

            if values:
                request = {"filetype": filetype,
                           "output_filepath": output_filepaths,
                           "column": column,
                           "value": values}
                talk.put(json.dumps(request), ttr=3600)
                log("Put {} requests into the queue".format(len(values)), INFO)

    talk.close()

def consumer(ip=IP_BEANSTALK, port=PORT_BEANSTALK, task=COMPETITION_GROUP_NAME, n_jobs=1):
    df_train = None
    df_train = pd.read_csv(TRAIN_FILE)
    log("Load {} completely".format(TRAIN_FILE))

    df_test = pd.read_csv(TEST_FILE)
    log("Load {} completely".format(TEST_FILE))

    week = 10
    median_route_solution = (load_median_solution(week-1, "route_id", ROUTE_GROUPS), ROUTE_GROUPS)
    median_agency_solution = (load_median_solution(week-1, "agency_id", AGENCY_GROUPS), AGENCY_GROUPS)

    talk = beanstalkc.Connection(host=ip, port=port)
    talk.watch(task)

    for n in range(0, n_jobs):
       thread = SplitThread(kwargs={"df_train": df_train, "df_test": df_test, "median_route_solution": median_route_solution, "median_agency_solution": median_agency_solution,"queue": queue})
       thread.setDaemon(True)
       thread.start()

    while True:
        job = talk.reserve(timeout=TIMEOUT_BEANSTALK)
        if job:
            o = json.loads(job.body)
            filetype, output_filepaths, column, values = o["filetype"], o["output_filepath"], o["column"], o["value"]

            output_folder = None
            for output_filepath, value in zip(output_filepaths, values):
                output_folder = os.path.dirname(output_filepath)

                create_folder(output_filepath)
                queue.put((output_filepath, filetype, column, value))

            queue.join()

            job.delete()

    queue.join()
    talk.close()

def feature_engineer_producer(columns, ip=IP_BEANSTALK, port=PORT_BEANSTALK, task=COMPETITION_FEATURE_ENGINEER_NAME):
    talk = beanstalkc.Connection(host=ip, port=port)
    talk.watch(task)

    files = os.path.join(SPLIT_PATH, "test", COLUMNS[column], "*.csv")
    for filepath_testing in glob.iglob(files):
        request = {"filepath": filepath_testing,
                   "column": column}

        talk.put(json.dumps(request), ttr=3600)
        log("Put {} requests into the queue".format(filepath_testing), INFO)

    talk.close()

def feature_engineer_consumer(ip=IP_BEANSTALK, port=PORT_BEANSTALK, task=COMPETITION_FEATURE_ENGINEER_NAME):
    talk = beanstalkc.Connection(host=ip, port=port)
    talk.watch(task)

    def history(df, column, week):
        lag = {"client": {},
               "median_channel": [],
               "median_column": []}

        # Semana,Agencia_ID,Canal_ID,Ruta_SAK,Cliente_ID,Producto_ID,Venta_uni_hoy,Venta_hoy,Dev_uni_proxima,Dev_proxima,Demanda_uni_equil
        for value in df[df[COLUMN_WEEK] == week].values:
            w, agency_id, channel_id, route_id, client_id, product_id, _, _, return_unit, _, unit = value
            return_unit = return_unit
            unit = unit

            lag["client"]["{}_{}".format(client_id, product_id)] = [return_unit, unit]
            lag["median_channel"]["{}_{}".format(channel_id, product_id)].append(unit)
            lag["median_column"]["{}_{}".format(agency_id if column.lower() == MONGODB_COLUMNS[COLUMN_AGENCY] else route_id, product_id)].append(unit)

        for key in ["median_channel_product", "median_column_product"]:
            for subkey, values in lag[key].items():
                lag[key][subkey] = np.median(values)

        return lag

    while True:
        job = talk.reserve(timeout=TIMEOUT_BEANSTALK)
        if job:
            o = json.loads(job.body)
            filepath_testing, column = o["filepath"], o["column"]
            filepath_training = filepath_testing.replace("test", "train")

            df_training = pd.read_csv(filepath_training)
            df_testing = pd.read_csv(filepath_testing)

            lag_1, lag2, lag3 = history(df_train, column, 5), history(df_train, column, 4), history(df_train, column, 3)
            lag_latest = {"client": {},
                          "median_channel": [],
                          "median_column": []}

            filepath_output = filepath_training.replace("split", "split.v2")
            create_folder(filepath_output)

            with open(filepath_output, "wb") as OUTPUT:
                OUTPUT.write("Semana,Agencia_ID,Canal_ID,Ruta_SAK,Cliente_ID,Producto_ID,Demanda_uni_equil,\
                             lag1_client_product,lag1_median_channel,lag1_median_column,\
                             lag2_client_product,lag2_median_channel,lag2_median_column,\
                             lag3_client_product,lag3_median_channel,lag3_median_column,\
                             return_1,return_2,return_3,trend_1,trend_2\n")

                for week in range(6, 9):
                    df = df_training[df_training[COLUMN_WEEK] == week]

                    for value in df.values:
                        w, agency_id, channel_id, route_id, client_id, product_id, _, _, return_unit, _, unit = value
                        return_unit = return_unit
                        unit = unit

                        key_client = "{}_{}".format(client_id, product_id)
                        key_channel = "{}_{}".format(channel_id, product_id)
                        key_column = "{}_{}".format(agency_id if column.lower() == MONGODB_COLUMNS[COLUMN_AGENCY] else route_id, product_id)

                        OUTPUT.write(",".join([str(w), str(agency_id), str(channel_id), str(route_id), str(client_id), str(product_id), str(unit),
                                               str(lag_1["client"].get(key_client, [0, 0])[1]), str(lag_1["median_channel"].get(key_channel, 0)), str(lag_1["median_column"].get(key_column, 0)),
                                               str(lag_2["client"].get(key_client, [0, 0])[1]), str(lag_2["median_channel"].get(key_channel, 0)), str(lag_2["median_column"].get(key_column, 0)),
                                               str(lag_3["client"].get(key_client, [0, 0])[1]), str(lag_3["median_channel"].get(key_channel, 0)), str(lag_3["median_column"].get(key_column, 0)),
                                               str(lag_1["client"].get(key_client, [0, 0])[0]), str(lag_2["client"].get(key_client, [0, 0])[0]), str(lag_3["client"].get(key_client, [0, 0])[0]),
                                               str(lag_1["client"].get(key_client, [0, 0])[1] - lag_2["client"].get(key_client, [0, 0])[1]),
                                               str(lag_2["client"].get(key_client, [0, 0])[1] - lag_3["client"].get(key_client, [0, 0])[1]), "\n"]))

                        lag_latest["client"]["{}_{}".format(client_id, product_id)] = [return_unit, unit]
                        lag_latest["median_channel"]["{}_{}".format(channel_id, product_id)].append(unit)
                        lag_latest["median_column"]["{}_{}".format(agency_id if column.lower() == MONGODB_COLUMNS[COLUMN_AGENCY] else route_id, product_id)].append(unit)

                    for key in ["median_channel_product", "median_column_product"]:
                        for subkey, values in lag_latest[key].items():
                            lag_latest[key][subkey] = np.median(values)

                    lag_3 = lag_2
                    lag_2 = lag_1
                    lag_1 = lag_latest

            filepath_output = filepath_testing.replace("split", "split.v2")
            create_folder(filepath_output)

            with open(filepath_output, "wb") as OUTPUT:
                OUTPUT.write("Semana,Agencia_ID,Canal_ID,Ruta_SAK,Cliente_ID,Producto_ID,Demanda_uni_equil,\
                             lag1_client_product,lag1_median_channel,lag1_median_column,\
                             lag2_client_product,lag2_median_channel,lag2_median_column,\
                             lag3_client_product,lag3_median_channel,lag3_median_column,\
                             return_1,return_2,return_3,trend_1,trend_2\n")

                for value in df_testing.values:
                    w, agency_id, channel_id, route_id, client_id, product_id = value

                    key_client = "{}_{}".format(client_id, product_id)
                    key_channel = "{}_{}".format(channel_id, product_id)
                    key_column = "{}_{}".format(agency_id if column.lower() == MONGODB_COLUMNS[COLUMN_AGENCY] else route_id, product_id)

                    OUTPUT.write(",".join([str(w), str(agency_id), str(channel_id), str(route_id), str(client_id), str(product_id), str(unit),
                                           str(lag_1["client"].get(key_client, [0, 0])[1]), str(lag_1["median_channel"].get(key_channel, 0)), str(lag_1["median_column"].get(key_column, 0)),
                                           str(lag_2["client"].get(key_client, [0, 0])[1]), str(lag_2["median_channel"].get(key_channel, 0)), str(lag_2["median_column"].get(key_column, 0)),
                                           str(lag_3["client"].get(key_client, [0, 0])[1]), str(lag_3["median_channel"].get(key_channel, 0)), str(lag_3["median_column"].get(key_column, 0)),
                                           str(lag_1["client"].get(key_client, [0, 0])[0]), str(lag_2["client"].get(key_client, [0, 0])[0]), str(lag_3["client"].get(key_client, [0, 0])[0]),
                                           str(lag_1["client"].get(key_client, [0, 0])[1] - lag_2["client"].get(key_client, [0, 0])[1]),
                                           str(lag_2["client"].get(key_client, [0, 0])[1] - lag_3["client"].get(key_client, [0, 0])[1]), "\n"]))

            job.delete()

    talk.close()

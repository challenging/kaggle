#!/usr/bin/env python

import os
import sys
import time
import glob

import threading
import Queue

import pymongo

import pandas as pd

from utils import log, INFO
from bimbo.constants import MONGODB_URL, MONGODB_DATABASE, MONGODB_STATS_COLLECTION, SPLIT_PATH, TEST_FILE

class BimboStatsThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        threading.Thread.__init__(self, group=group, target=target, name=name, verbose=verbose)

        self.args = args

        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def get_queue(filepath=TEST_FILE):
        queue = Queue.Queue()

        df = pd.read_csv(filepath)
        count = df.shape[0]

        batch_num = 5000
        for idx in range(0, count/batch_num+1):
            idx_start = idx*batch_num
            idx_end = min(count, (idx+1)*batch_num)

            queue.put(df.values[idx_start:idx_end])

        return queue

    def run(self):
        client = pymongo.MongoClient(MONGODB_URL)
        collection = client[MONGODB_DATABASE][MONGODB_STATS_COLLECTION]
        collection.create_index([("agency_id",pymongo.ASCENDING)])
        collection.create_index([("channel_id",pymongo.ASCENDING)])
        collection.create_index([("route_id",pymongo.ASCENDING)])
        collection.create_index([("client_id",pymongo.ASCENDING)])
        collection.create_index([("product_id",pymongo.ASCENDING)])

        while True:
            values = self.queue.get()

            records = []
            for record in stats1(self.df_train, values):
                records.append(record)

            if records:
                log(collection.insert_many(records), INFO)

            self.queue.task_done()

        client.close()

def stats1(df_train, values):
    for value in values:
        timestamp_start = time.time()

        row_id, week_num, agency_id, channel_id, route_id, client_id, product_id = value
        record = {
                    "row_id": row_id,
                    "week_num": week_num,
                    "agency_id": agency_id,
                    "channel_id": channel_id,
                    "route_id": route_id,
                    "client_id": client_id,
                    "product_id": product_id,
                 }

        count_1_0, count_1_1, count_1_2, count_1_3 = 0, 0, 0, 0
        count_2_0, count_2_1, count_2_2, count_2_3, count_2_4, count_2_5 = 0, 0, 0, 0, 0, 0
        count_3_0, count_3_1, count_3_2, count_3_3 = 0, 0, 0, 0
        count_4 = 0

        # Lookup 4-dimensions
        count_4 = self.df_train[(self.df_train["Producto_ID"] == product_id) &
                                (self.df_train["Agencia_ID"] == agency_id) &
                                (self.df_train["Canal_ID"] == channel_id) &
                                (self.df_train["Ruta_SAK"] == route_id) &
                                (self.df_train["Cliente_ID"] == client_id)].shape[0]

        if count_4 > 0:
            record["4_dimension"] = [{"agency_channel_route_client": count_4}]

        # Lookup 3-dimensions
        if count_4 == 0:
            count_3_0 = self.df_train[(self.df_train["Producto_ID"] == product_id) &
                                      (self.df_train["Agencia_ID"] == agency_id) &
                                      (self.df_train["Canal_ID"] == channel_id) &
                                      (self.df_train["Ruta_SAK"] == route_id)].shape[0]
            if count_3_0 > 0:
                record.setdefault("3_dimension", [])
                record["3_dimension"].append({"agency_channel_route": count_3_0})

            count_3_1 = self.df_train[(self.df_train["Producto_ID"] == product_id) &
                                      (self.df_train["Agencia_ID"] == agency_id) &
                                      (self.df_train["Canal_ID"] == channel_id) &
                                      (self.df_train["Cliente_ID"] == client_id)].shape[0]
            if count_3_1 > 0:
                record.setdefault("3_dimension", [])
                record["3_dimension"].append({"agency_channel_client": count_3_1})

            count_3_2 = self.df_train[(self.df_train["Producto_ID"] == product_id) &
                                      (self.df_train["Agencia_ID"] == agency_id) &
                                      (self.df_train["Cliente_ID"] == client_id) &
                                      (self.df_train["Ruta_SAK"] == route_id)].shape[0]
            if count_3_2 > 0:
                record.setdefault("3_dimension", [])
                record["3_dimension"].append({"agency_channel_route": count_3_2})

            count_3_3 = self.df_train[(self.df_train["Producto_ID"] == product_id) &
                                      (self.df_train["Canal_ID"] == channel_id) &
                                      (self.df_train["Ruta_SAK"] == route_id) &
                                      (self.df_train["Cliente_ID"] == client_id)].shape[0]
            if count_3_3 > 0:
                record.setdefault("3_dimension", [])
                record["3_dimension"].append({"channel_route_client": count_3_3})

            # Lookup 2-dimension
            if count_3_0 + count_3_1 + count_3_2 + count_3_3 == 0:
                count_2_0 = self.df_train[(self.df_train["Producto_ID"] == product_id) &
                                          (self.df_train["Agencia_ID"] == agency_id) &
                                          (self.df_train["Canal_ID"] == channel_id)].shape[0]
                if count_2_0 > 0:
                    record.setdefault("2_dimension", [])
                    record["2_dimension"].append({"agency_channel": count_2_0})

                count_2_1 = self.df_train[(self.df_train["Producto_ID"] == product_id) &
                                          (self.df_train["Agencia_ID"] == agency_id) &
                                          (self.df_train["Ruta_SAK"] == route_id)].shape[0]
                if count_2_1 > 0:
                    record.setdefault("2_dimension", [])
                    record["2_dimension"].append({"agency_route": count_2_1})

                count_2_2 = self.df_train[(self.df_train["Producto_ID"] == product_id) &
                                          (self.df_train["Agencia_ID"] == agency_id) &
                                          (self.df_train["Cliente_ID"] == client_id)].shape[0]
                if count_2_2 > 0:
                    record.setdefault("2_dimension", [])
                    record["2_dimension"].append({"agency_client": count_2_2})

                count_2_3 = self.df_train[(self.df_train["Producto_ID"] == product_id) &
                                          (self.df_train["Canal_ID"] == channel_id) &
                                          (self.df_train["Ruta_SAK"] == route_id)].shape[0]
                if count_2_3 > 0:
                    record.setdefault("2_dimension", [])
                    record["2_dimension"].append({"channel_route": count_2_3})

                count_2_4 = self.df_train[(self.df_train["Producto_ID"] == product_id) &
                                          (self.df_train["Canal_ID"] == channel_id) &
                                          (self.df_train["Cliente_ID"] == client_id)].shape[0]
                if count_2_4 > 0:
                    record.setdefault("2_dimension", [])
                    record["2_dimension"].append({"channel_client": count_2_4})

                count_2_5 = self.df_train[(self.df_train["Producto_ID"] == product_id) &
                                          (self.df_train["Ruta_SAK"] == route_id) &
                                          (self.df_train["Cliente_ID"] == client_id)].shape[0]
                if count_2_5 > 0:
                    record.setdefault("2_dimension", [])
                    record["2_dimension"].append({"route_client": count_2_5})

                # Lookup 1-dimension
                if count_2_0 + count_2_1 + count_2_2 + count_2_3 + count_2_4 + count_2_5 == 0:
                    record["1_dimension"] = []

                    count_1_0 = self.df_train[(self.df_train["Producto_ID"] == product_id) & (self.df_train["Agencia_ID"] == agency_id)].shape[0]
                    if count_1_0 > 0:
                        record.setdefault("1_dimension", [])
                        record["1_dimension"].append({"agency": count_1_0})

                    count_1_1 = self.df_train[(self.df_train["Producto_ID"] == product_id) & (self.df_train["Canal_ID"] == channel_id)].shape[0]
                    if count_1_1 > 0:
                        record.setdefault("1_dimension", [])
                        record["1_dimension"].append({"channel": count_1_1})

                    count_1_2 = self.df_train[(self.df_train["Producto_ID"] == product_id) & (self.df_train["Ruta_SAK"] == route_id)].shape[0]
                    if count_1_2 > 0:
                        record.setdefault("1_dimension", [])
                        record["1_dimension"].append({"route": count_1_2})

                    count_1_3 = self.df_train[(self.df_train["Producto_ID"] == product_id) & (self.df_train["Cliente_ID"] == client_id)].shape[0]
                    if count_1_3 > 0:
                        record.setdefault("1_dimension", [])
                        record["1_dimension"].append({"client": count_1_3})

        timestamp_end = time.time()
        log("Cost {:4f} secends to query {} ID({}/{}/{}/{}/{}/{}/{}/{}/{}/{}/{}/{}/{}/{}/{})".format(timestamp_end-timestamp_start, row_id,\
                count_1_0, count_1_1, count_1_2, count_1_3,\
                count_2_0, count_2_1, count_2_2, count_2_3, count_2_4, count_2_5,\
                count_3_0, count_3_1, count_3_2, count_3_3,\
                count_4), INFO)

        yield record

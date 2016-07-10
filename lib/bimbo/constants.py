#!/usr/bin/env python

import os

COMPETITION_NAME = "bimbo_competition"

WORKSPACE = "/Users/rongqichen/Documents/programs/kaggle/cases/Grupo Bimbo Inventory Demand"
DATA_PATH = os.path.join(WORKSPACE, "input")
SPLIT_PATH = os.path.join(DATA_PATH, "split")

TRAIN_FILE = os.path.join(DATA_PATH, "train.csv")
TEST_FILE = os.path.join(DATA_PATH, "test.csv")

TESTING_TRAIN_FILE = os.path.join(DATA_PATH, "train_10000.csv")
TESTING_TEST_FILE = os.path.join(DATA_PATH, "test_10000.csv")

COLUMN_AGENCY, COLUMN_CHANNEL, COLUMN_ROUTE, COLUMN_PRODUCT, COLUMN_CLIENT = "Agencia_ID", "Canal_ID", "Ruta_SAK", "Producto_ID", "Cliente_ID"
COLUMNS = {"agency_id": COLUMN_AGENCY,
           "channel_id": COLUMN_CHANNEL,
           "route_id": COLUMN_ROUTE,
           "product_id": COLUMN_PRODUCT,
           "client_id": COLUMN_CLIENT}

IP_BEANSTALK, PORT_BEANSTALK = "rongqide-Mac-mini.local", 11300
#IP_BEANSTALK = "sakaes-MacBook-Pro.local"
TIMEOUT_BEANSTALK=1800
TASK_BEANSTALK = "bimbo_competition"

MONGODB_URL = "mongodb://{}:27017".format(IP_BEANSTALK)
MONGODB_BATCH_SIZE = 20000

MONGODB_DATABASE, MONGODB_STATS_COLLECTION = "bimbo", "naive_stats"
MONGODB_COLUMNS = {COLUMN_AGENCY: "agency_id",
                   COLUMN_CHANNEL: "channel_id",
                   COLUMN_ROUTE: "route_id",
                   COLUMN_PRODUCT: "product_id",
                   COLUMN_CLIENT: "client_id"}

def get_stats_mongo_collection(name):
    return "{}_{}".format(MONGODB_STATS_COLLECTION, name.lower())

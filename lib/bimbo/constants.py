#!/usr/bin/env python

import os
import pymongo

PYPY = "/Users/RungChiChen/Downloads/pypy2-v5.3.1-osx64/bin/pypy"

COMPETITION_NAME = "bimbo_competition"
COMPETITION_GROUP_NAME = "{}_stats".format(COMPETITION_NAME)
COMPETITION_CC_NAME = "{}_cc".format(COMPETITION_NAME)

NON_PREDICTABLE = -1

WORKSPACE = "/Users/rongqichen/Documents/programs/kaggle/cases/Grupo Bimbo Inventory Demand"
MEDIAN_SOLUTION_PATH = os.path.join(WORKSPACE, "median_solution")
FTLR_SOLUTION_PATH = os.path.join(WORKSPACE, "ftlr_solution")

DATA_PATH = os.path.join(WORKSPACE, "input")
SPLIT_PATH = os.path.join(DATA_PATH, "split")
STATS_PATH = os.path.join(DATA_PATH, "stats")

TRAIN_FILE = os.path.join(DATA_PATH, "train.csv")
TEST_FILE = os.path.join(DATA_PATH, "test.csv")

TESTING_TRAIN_FILE = os.path.join(DATA_PATH, "10000", "train_10000.csv")
TESTING_TEST_FILE = os.path.join(DATA_PATH, "10000", "test_10000.csv")

COLUMN_WEEK, COLUMN_ROW = "Semana", "row_id"
COLUMN_AGENCY, COLUMN_CHANNEL, COLUMN_ROUTE, COLUMN_PRODUCT, COLUMN_CLIENT = "Agencia_ID", "Canal_ID", "Ruta_SAK", "Producto_ID", "Cliente_ID"
COLUMN_PREDICTION = "Demanda_uni_equil"
COLUMNS = {"agency_id": COLUMN_AGENCY,
           "channel_id": COLUMN_CHANNEL,
           "route_id": COLUMN_ROUTE,
           "product_id": COLUMN_PRODUCT,
           "client_id": COLUMN_CLIENT,
           "week": COLUMN_WEEK}

BATCH_JOB = 5000

IP_BEANSTALK, PORT_BEANSTALK = "rongqide-Mac-mini.local", 11300
#IP_BEANSTALK = "sakaes-MacBook-Pro.local"
TIMEOUT_BEANSTALK=3600*3
TASK_BEANSTALK = "bimbo_competition"

MONGODB_URL = "mongodb://{}:27017".format(IP_BEANSTALK)
MONGODB_BATCH_SIZE = 20000

MONGODB_DATABASE, MONGODB_STATS_COLLECTION, MONGODB_STATS_CC_COLLECTION = "bimbo", "naive_stats", "cc_stats"
MONGODB_COLUMNS = {COLUMN_AGENCY: "agency_id",
                   COLUMN_CHANNEL: "channel_id",
                   COLUMN_ROUTE: "route_id",
                   COLUMN_PRODUCT: "product_id",
                   COLUMN_CLIENT: "client_id"}

def get_stats_mongo_collection(name):
    return "{}_{}".format(MONGODB_STATS_COLLECTION, name.lower())

def get_mongo_connection():
    return pymongo.MongoClient(MONGODB_URL)

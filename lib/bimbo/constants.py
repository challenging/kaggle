#!/usr/bin/env python

import os
import json
import pymongo

from utils import log, INFO

TOTAL_WEEK = 12

PYPY = "/Users/rongqichen/Documents/programs/kaggle/github/bin/pypy2-v5.3.1-osx64/bin/pypy"

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

TESTING_TRAIN_FILE = os.path.join(DATA_PATH, "testing", "train.csv")
TESTING_TEST_FILE = os.path.join(DATA_PATH, "testing", "test.csv")

COLUMN_WEEK, COLUMN_ROW = "Semana", "row_id"
COLUMN_AGENCY, COLUMN_CHANNEL, COLUMN_ROUTE, COLUMN_PRODUCT, COLUMN_CLIENT = "Agencia_ID", "Canal_ID", "Ruta_SAK", "Producto_ID", "Cliente_ID"
COLUMN_PREDICTION = "Demanda_uni_equil"
COLUMNS = {"agency_id": COLUMN_AGENCY,
           "channel_id": COLUMN_CHANNEL,
           "route_id": COLUMN_ROUTE,
           "product_id": COLUMN_PRODUCT,
           "client_id": COLUMN_CLIENT,
           "week": COLUMN_WEEK}

AGENCY_GROUPS = [[COLUMN_AGENCY, COLUMN_PRODUCT, COLUMN_CLIENT],
                [COLUMN_PRODUCT, COLUMN_CLIENT],
                [COLUMN_AGENCY, COLUMN_PRODUCT],
                [COLUMN_AGENCY, COLUMN_CLIENT],
                [COLUMN_PRODUCT]]

ROUTE_GROUPS = [[COLUMN_ROUTE, COLUMN_PRODUCT, COLUMN_CLIENT],
                [COLUMN_PRODUCT, COLUMN_CLIENT],
                [COLUMN_ROUTE, COLUMN_PRODUCT],
                [COLUMN_ROUTE, COLUMN_CLIENT],
                [COLUMN_PRODUCT]]

BATCH_JOB = 5000

IP_BEANSTALK, PORT_BEANSTALK = "rongqide-Mac-mini.local", 11300
#IP_BEANSTALK = "rongqis-iMac.local"
TIMEOUT_BEANSTALK=3600*3
TASK_BEANSTALK = "bimbo_competition"

MONGODB_URL = "mongodb://{}:27017".format(IP_BEANSTALK)
MONGODB_BATCH_SIZE = 20000

MONGODB_DATABASE, MONGODB_PREDICTION_DATABASE, MONGODB_CC_DATABASE = "bimbo", "bimbo_prediction", "bimbo_cc"
MONGODB_STATS_COLLECTION, MONGODB_STATS_CC_COLLECTION, MONGODB_PREDICTION_COLLECTION = "naive_stats", "cc_stats", "prediction"
MONGODB_COLUMNS = {COLUMN_AGENCY: "agency_id",
                   COLUMN_CHANNEL: "channel_id",
                   COLUMN_ROUTE: "route_id",
                   COLUMN_PRODUCT: "product_id",
                   COLUMN_CLIENT: "client_id",
                   COLUMN_WEEK: "week"}

def get_stats_mongo_collection(name):
    return "{}_{}".format(MONGODB_STATS_COLLECTION, name).lower()

def get_cc_mongo_collection(name):
    return "cc_{}".format(name).lower()

def get_prediction_mongo_collection(name):
    return "{}_{}".format(MONGODB_PREDICTION_COLLECTION, name).lower()

def get_mongo_connection():
    return pymongo.MongoClient(MONGODB_URL)

def load_median_solution(week, filetype, groups):
    solutions = []

    for group in groups:
        filepath = os.path.join(MEDIAN_SOLUTION_PATH, filetype, "week={}".format(week), "{}.json".format("_".join(group)))

        log("Start to read median solution from {}".format(filepath), INFO)
        with open(filepath, "rb") as INPUT:
            solution = json.load(INPUT)

        solutions.append(solution)

    return solutions

def get_median(classifiers, keys, values):
    for key, classifier in zip(keys, classifiers):
        k = "_".join([str(values[k]) for k in key])

        if k in classifier:
            return classifier[k]

    return 0.0

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

from utils import log, create_folder
from utils import DEBUG, INFO, WARN
from bimbo.constants import get_stats_mongo_collection, get_mongo_connection
from bimbo.constants import IP_BEANSTALK, MONGODB_DATABASE, MONGODB_BATCH_SIZE, COLUMN_ROW, MONGODB_COLUMNS, COLUMNS, SPLIT_PATH

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

def hierarchical_folder_structure(column, filetype):
    prefixs = set()
    folder = os.path.join(SPLIT_PATH, COLUMNS[column], filetype.lower())

    timestamp_start = time.time()
    for filepath in glob.iglob("{}/*.csv".format(folder)):
        filename = os.path.basename(filepath)

        prefix = filename[0:3]
        prefixs.add(prefix)

        new_folder = os.path.join(folder, prefix)
        new_filepath = os.path.join(new_folder, filename)

        create_folder(new_filepath)
        os.rename(filepath, new_filepath)
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

@click.command()
@click.option("--column", default=None, help="agency_id|channel_id|route_id|client_id|product_id")
@click.option("--mode", required=True, help="purge|restructure")
def tool(column, mode):
    if mode == "purge":
        purge_duplicated_records(column)
    elif mode == "restructure":
        for filetype in ["train", "test"]:
            hierarchical_folder_structure(column, filetype)

if __name__ ==  "__main__":
    tool()

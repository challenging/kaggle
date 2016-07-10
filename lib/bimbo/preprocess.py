#!/usr/bin/env python

import os
import sys
import time
import click

import threading
import Queue

import pandas as pd

from utils import create_folder, log, INFO
from bimbo.constants import COLUMNS

WORKSPACE = "/Users/rongqichen/Documents/programs/kaggle/cases/Grupo Bimbo Inventory Demand/input"
TRAIN_FILE = os.path.join(WORKSPACE, "train.csv")
TEST_FILE = os.path.join(WORKSPACE, "test.csv")

# Semana,Agencia_ID,Canal_ID,Ruta_SAK,Cliente_ID,Producto_ID,Venta_uni_hoy,Venta_hoy,Dev_uni_proxima,Dev_proxima,Demanda_uni_equil

class SplitThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        threading.Thread.__init__(self, group=group, target=target, name=name, verbose=verbose)

        self.args = args

        for key, value in kwargs.items():
            setattr(self, key, value)

    def run(self):
        while True:
            timestamp_start = time.time()

            output_filepath, column, value = self.queue.get()
            if os.path.exists(output_filepath):
                log("Found {}, so skipping".format(output_filepath), INFO)
            else:
                row_ids = self.df[column] == value
                self.df[row_ids].to_csv(output_filepath, index=False)

            self.queue.task_done()

            timestamp_end = time.time()
            log("Cost {:4}f secends to store {}={} in {}, the remaining queue size is {}".format(timestamp_end-timestamp_start, column, value, output_filepath, self.queue.qsize()), INFO)

@click.command()
@click.option("--columns", required=True, help="column name for split")
@click.option("--n-jobs", default=1, help="number of thread")
def preprocess(columns, n_jobs):
    global TRAIN_FILE, TEST_FILE

    queue = Queue.Queue()
    for filepath in [TRAIN_FILE, TEST_FILE]:
        df = pd.read_csv(filepath)

        for column in columns.split(","):
            column = COLUMNS[column]

            output_folder = os.path.join(WORKSPACE, "split", column, os.path.basename(filepath).replace(".csv", ""))
            create_folder(os.path.join(output_folder, "1.txt"))

            for n in range(0, n_jobs):
                thread = SplitThread(kwargs={"df": df, "queue": queue})
                thread.setDaemon(True)
                thread.start()

            for unique_value in df[column].unique():
                output_filepath = os.path.join(output_folder, "{}.csv".format(unique_value))
                queue.put((output_filepath, column, unique_value))

            queue.join()

if __name__ == "__main__":
    preprocess()

#!/usr/bin/env python

import os
import sys
import json
import time
import click
import socket
import subprocess
import beanstalkc

import threading
import Queue

import pandas as pd

from utils import create_folder, log, INFO, WARN
from bimbo.constants import COLUMNS, COMPETITION_GROUP_NAME, IP_BEANSTALK, PORT_BEANSTALK, TIMEOUT_BEANSTALK

WORKSPACE = "/Users/rongqichen/Documents/programs/kaggle/cases/Grupo Bimbo Inventory Demand/input"
TRAIN_FILE = os.path.join(WORKSPACE, "train.csv")
TEST_FILE = os.path.join(WORKSPACE, "test.csv")

class SplitThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        threading.Thread.__init__(self, group=group, target=target, name=name, verbose=verbose)

        self.args = args

        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def matching(df, column, value):
        return df[df[column] == value]

    def run(self):
        hostname = socket.gethostname()

        while True:
            timestamp_start = time.time()

            output_filepath, filetype, column, value, job = self.queue.get()

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
                        raise NotImplementError
                else:
                    raise NotImplementError

                self.matching(df, column, value).to_csv(output_filepath, index=False)

                if hostname != IP_BEANSTALK:
                    p = subprocess.Popen(["scp", output_filepath, "RungChiChen@{}:{}".format(IP_BEANSTALK, output_filepath)])
                    sts = os.waitpid(p.pid, 0)
                    log("Transfer {} successfully({})".format(output_filepath, sts), INFO)


            if job:
                try:
                    job.delete()
                except Exception as e:
                    log(e, WARN)

            self.queue.task_done()

            timestamp_end = time.time()
            log("Cost {:4f} secends to store {}={} in {}, the remaining queue size is {}".format(timestamp_end-timestamp_start, column, value, output_filepath, self.queue.qsize()), INFO)

def producer(columns, ip=IP_BEANSTALK, port=PORT_BEANSTALK, task=COMPETITION_GROUP_NAME):
    global WORKSPACE, TRAIN_FILE, TEST_FILE

    talk = beanstalkc.Connection(host=ip, port=port)
    talk.watch(task)

    for filetype, filepath in zip(["train", "test"], [TRAIN_FILE, TEST_FILE]):
        log("Start to read {}".format(filepath), INFO)
        df = pd.read_csv(filepath)

        for column in columns.split(","):
            column = COLUMNS[column]

            output_folder = os.path.join(WORKSPACE, "split", column, os.path.basename(filepath).replace(".csv", ""))

            for unique_value in df[column].unique():
                output_filepath = os.path.join(output_folder, "{}.csv".format(unique_value))

                if os.path.exists(output_filepath):
                    log("Found {} so skipping it".format(output_filepath), INFO)
                else:
                    request = {"filetype": filetype,
                               "output_filepath": output_filepath,
                               "column": column,
                               "value": unique_value}
                    talk.put(json.dumps(request), TTR=3600)

                    log("Put {} into the queue".format(request), INFO)

    talk.close()

def consumer(ip=IP_BEANSTALK, port=PORT_BEANSTALK, task=COMPETITION_GROUP_NAME, n_jobs=1):
    global WORKSPACE, TRAIN_FILE, TEST_FILE
    df_train = pd.read_csv(TRAIN_FILE)
    df_test = pd.read_csv(TEST_FILE)

    talk = beanstalkc.Connection(host=ip, port=port)
    talk.watch(task)

    queue = Queue.Queue()
    for n in range(0, n_jobs):
       thread = SplitThread(kwargs={"df_train": df_train, "df_test": df_test, "queue": queue})
       thread.setDaemon(True)
       thread.start()

    while True:
        if queue.qsize() > 128:
            log("Start to sleep 60 secends", INFO)
            time.sleep(60)
        else:
            job = talk.reserve(timeout=TIMEOUT_BEANSTALK)
            if job:
                o = json.loads(job.body)
                filetype, output_filepath, column, value = o["filetype"], o["output_filepath"], o["column"], o["value"]

                create_folder(output_filepath)

                queue.put((output_filepath, filetype, column, value, job))

    queue.join()
    talk.close()

@click.command()
@click.option("--mode", default=None, help="producer mode | consumer mode when enable beanstalk")
@click.option("--columns", default=None, help="column name for split")
@click.option("--n-jobs", default=1, help="number of thread")
def preprocess(mode, columns, n_jobs):
    global WORKSPACE, TRAIN_FILE, TEST_FILE

    if mode:
        if mode.lower() == "producer":
            producer(columns)
        else:
            consumer(n_jobs=n_jobs)
    else:
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

                    if os.path.exists(output_filepath):
                        log("Found {} so skipping it".format(output_filepath), INFO)
                    else:
                        queue.put((output_filepath, None, column, unique_value, None))
                        log("Put {} into the queue".format(output_filepath), INFO)

                queue.join()

if __name__ == "__main__":
    preprocess()

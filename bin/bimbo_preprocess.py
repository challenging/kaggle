#!/usr/bin/env python

import os
import sys
import click
import Queue

from bimbo.preprocess import SplitThread
from bimbo.constants import SPLIT_PATH, TRAIN_FILE, TEST_FILE

@click.command()
@click.option("--mode", default=None, help="producer mode | consumer mode when enable beanstalk")
@click.option("--columns", default=None, help="column name for split")
@click.option("--n-jobs", default=1, help="number of thread")
def preprocess(mode, columns, n_jobs):
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

                output_folder = os.path.join(SPLIT_PATH, column, os.path.basename(filepath).replace(".csv", ""))
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

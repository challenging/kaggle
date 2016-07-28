#!/usr/bin/env python

import click

from bimbo.preprocess import producer, consumer, feature_engineer_producer, feature_engineer_consumer
from bimbo.constants import SPLIT_PATH

@click.command()
@click.option("--option", required=True, nargs=2, type=click.Tuple([unicode, unicode]), default=(None, None), help="producer mode | consumer mode")
@click.option("--column", default=None, help="column name for split")
@click.option("--n-jobs", default=1, help="number of thread")
def preprocess(mode, column, n_jobs):
    solution, mode = option

    solution = solution.lower()
    mode = mode.lower()

    if solution == "split":
        if mode == "producer":
            producer(column)
        elif mode == "consumer":
            consumer(n_jobs=n_jobs)
        else:
            raise NotImplementedError
    elif solution == "feature_engineer":
        if mode == "producer":
            feature_engineer_producer(column)
        elif mode == "consumer":
            feature_engineer_consumer(column)
        else:
            raise NotImplementedError

if __name__ == "__main__":
    preprocess()

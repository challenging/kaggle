#!/usr/bin/env python

import os
import sys
import glob
import click
import threading

from utils import log, INFO
from bimbo.constants import load_median_route_solution, COMPETITION_CC_NAME, COLUMNS, ROUTE_GROUPS, SPLIT_PATH

@click.command()
@click.option("--is-testing", is_flag=True, help="testing mode")
@click.option("--n-jobs", default=1, help="number of thread")
@click.option("--week", default=9, help="week number(6-9)")
@click.option("--column", default="agency_id", help="agency_id|channel_id|route_id|product_id")
@click.option("--option", required=True, nargs=2, type=click.Tuple([unicode, unicode]), default=(None, None), help="producer mode | consumer mode")
def stats(is_testing, n_jobs, week, column, option):
    beanstype, mode = option

    task = COMPETITION_CC_NAME

    if beanstype == "stats":
        from bimbo.stats_beanstalk import producer, consumer

        if mode == "producer":
            producer(column, is_testing, ttr=86400*3)
        elif mode == "consumer":
            consumer()
        else:
            log("Not implement this mode({})".format(mode), INFO)

            sys.exit(999)
    elif beanstype == "cc":
        from bimbo.cc_beanstalk import producer, consumer

        task += "_{}_{}".format(column, week)

        if mode.lower() == "producer":
            count = 1
            pattern_file = os.path.join(SPLIT_PATH, COLUMNS[column], "train", "*.csv")
            for filepath in glob.iglob(pattern_file):
                filename = os.path.basename(filepath)
                fid = filename.replace(".csv", "")

                producer(week, (column, fid), task=task)

                count += 1
                if is_testing and count > 1:
                    break
        elif mode.lower() == "consumer":
            median_solution = ({}, {})

            if column == "route_id":
                median_solution = (load_median_route_solution(week-1), ROUTE_GROUPS)
            else:
                raise NotImplementError

            if is_testing:
                n_jobs = 1

            threads = []
            for i in range(0, n_jobs):
                thread = threading.Thread(target=consumer, kwargs={"median_solution": median_solution, "task": task})
                thread.setDaemon(True)
                thread.start()

                threads.append(thread)

            for thread in threads:
                thread.join()
        else:
            log("Not implement this mode({})".format(mode), INFO)

            sys.exit(999)

if __name__ == "__main__":
    stats()

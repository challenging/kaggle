#!/usr/bin/env python

import os
import sys
import glob
import click

from utils import log, create_folder
from utils import INFO

from joblib import Parallel, delayed

from utils import log, create_folder
from utils import DEBUG, INFO, WARN
from bimbo.constants import load_median_solution
from bimbo.constants import MONGODB_COLUMNS, COLUMNS, COLUMN_ROUTE, MONGODB_PREDICTION_DATABASE
from bimbo.constants import SPLIT_PATH, STATS_PATH, TRAIN_FILE, TEST_FILE, TESTING_TRAIN_FILE, TESTING_TEST_FILE, FTLR_SOLUTION_PATH, MEDIAN_SOLUTION_PATH
from bimbo.constants import ROUTE_GROUPS, AGENCY_GROUPS
from bimbo.constants import TRAIN_FILE, TEST_FILE, TESTING_TRAIN_FILE, TESTING_TEST_FILE
from bimbo.solutions import ensemble_solution, median_solution, ftlr_solution, cache_median
from bimbo.tools import purge_duplicated_records, hierarchical_folder_structure, repair_missing_records, aggregation, cc_solution

TRAIN = TRAIN_FILE
TEST = TEST_FILE

@click.command()
@click.option("--is-testing", is_flag=True, help="testing mode")
@click.option("--column", default=None, help="agency_id|channel_id|route_id|client_id|product_id")
@click.option("--mode", required=True, help="purge|restructure")
@click.option("--week", default=9, help="week number(4-9)")
@click.option("--option", required=False, nargs=2, type=click.Tuple([unicode, unicode]), default=(None, None))
def tool(is_testing, column, mode, week, option):
    global TRAIN, TEST

    if is_testing:
        TRAIN = TESTING_TRAIN_FILE
        TEST = TESTING_TEST_FILE

    if mode == "purge":
        purge_duplicated_records(column)
    elif mode == "restructure":
        for filetype in ["train", "test"]:
            hierarchical_folder_structure(column, filetype)
    elif mode == "repair":
        repair_missing_records(column)
    elif mode == "aggregation":
        columns = [COLUMNS[c] for c in column.split(",")]
        output_filepath = os.path.join(STATS_PATH, "{}.csv".format("_".join(columns)))
        create_folder(output_filepath)

        aggregation(TRAIN, columns, output_filepath)
    elif mode == "cc":
        column, column_value = option
        column_value = int(column_value)

        filepath = os.path.join(SPLIT_PATH, COLUMNS[column], "train", "{}.csv".format(column_value))

        solution = ([], [])
        if not is_testing:
            groups = None

            if MONGODB_COLUMNS[COLUMN_ROUTE] == column:
                groups = ROUTE_GROUPS
            elif MONGODB_COLUMNS[COLUMN_AGENCY] == column:
                groups = AGENCY_GROUPS

            solution = (load_median_solution(week-1, column), groups)

        cc_solution(week, filepath, filepath, (COLUMNS[column], column_value), solution)
    elif mode == "cache":
        cache_median(TRAIN, column, week)
    elif mode == "solution":
        solution, column = option

        if solution == "ftlr":
            folder = os.path.join(SPLIT_PATH, COLUMNS[column], "test")
            submission_folder = os.path.join(FTLR_SOLUTION_PATH, COLUMNS[column])
            create_folder("{}/1.txt".format(submission_folder))

            Parallel(n_jobs=6)(delayed(ftlr_solution)(folder, os.path.basename(filepath).replace(".csv", ""), submission_folder) for filepath in glob.iglob(os.path.join(folder, "*.csv")))
        elif solution == "median":
            output_filepath = os.path.join(MEDIAN_SOLUTION_PATH, "{}.csv.gz".format(column))
            filepath_test = os.path.join(SPLIT_PATH, COLUMNS[column], "test", "*.csv")

            groups = None
            if MONGODB_COLUMNS[COLUMN_ROUTE] == column:
                groups = ROUTE_GROUPS
            elif MONGODB_COLUMNS[COLUMN_ROUTE] == column:
                groups = AGENCY_GROUPS

            solution = (load_median_solution(column, week-1), groups)

            median_solution(output_filepath, filepath_test, solution)
    elif mode == "ensemble":
        filepaths, output_filepath = option

        ensemble_solution(filepaths.split(","), output_filepath)
    else:
        log("Not found this mode {}".format(mode), ERROR)
        sys.exit(101)

if __name__ ==  "__main__":
    tool()

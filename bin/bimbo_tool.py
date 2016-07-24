#!/usr/bin/env python

import os
import sys
import time
import glob
import click

from utils import log, create_folder
from utils import INFO

from joblib import Parallel, delayed

from utils import log, create_folder
from utils import DEBUG, INFO, WARN
from bimbo.constants import load_median_solution
from bimbo.constants import MONGODB_COLUMNS, COLUMNS, COLUMN_ROUTE, COLUMN_AGENCY, MONGODB_PREDICTION_DATABASE
from bimbo.constants import SPLIT_PATH, STATS_PATH, TRAIN_FILE, TEST_FILE, TESTING_TRAIN_FILE, TESTING_TEST_FILE, FTLR_SOLUTION_PATH, MEDIAN_SOLUTION_CACHE, MEDIAN_SOLUTION_PATH, CC_SOLUTION_PATH
from bimbo.constants import ROUTE_GROUPS, AGENCY_GROUPS
from bimbo.constants import TRAIN_FILE, TEST_FILE, TESTING_TRAIN_FILE, TESTING_TEST_FILE
from bimbo.solutions import ensemble_solution, median_solution, ftlr_solution, regression_solution, cache_median
from bimbo.tools import purge_duplicated_records, hierarchical_folder_structure, repair_missing_records, aggregation, cc_solution

TRAIN = TRAIN_FILE
TEST = TEST_FILE

def cc(column, column_value, week, is_output):
    ts = time.time()

    filepath = os.path.join(SPLIT_PATH, COLUMNS[column], "train", "{}.csv".format(column_value))
    if is_output:
        log("Turn on the output mode", INFO)

        filepath_output = os.path.join(CC_SOLUTION_PATH, "train", "week={}".format(week), COLUMNS[column], "submission_{}.csv".format(column_value))
        create_folder(filepath_output)

        if os.path.exists(filepath_output):
           log("Found {} so skipping it".format(filepath_output), INFO)
        else:
            with open(filepath_output, "wb") as OUTPUT:
                OUTPUT.write("Semana,Agencia_ID,Canal_ID,Ruta_SAK,Cliente_ID,Producto_ID,CC_Demanda_uni_equil\n")

                for row in cc_solution(week, filepath, filepath, (COLUMNS[column], column_value)):
                    OUTPUT.write("{}\n".format(row))
    else:
        log("Turn off the output mode", INFO)

        for row in cc_solution(week, filepath, filepath, (COLUMNS[column], column_value)):
            pass

    te = time.time()
    log("Cost {:4f} secends to finish {}".format(te-ts, filepath), INFO)

@click.command()
@click.option("--n-jobs", default=1, help="number of thread")
@click.option("--is-testing", is_flag=True, help="testing mode")
@click.option("--column", default=None, help="agency_id|channel_id|route_id|client_id|product_id")
@click.option("--mode", required=True, help="purge|restructure")
@click.option("--week", default=9, help="week number(4-9)")
@click.option("--is-output", is_flag=True, help="output mode")
@click.option("--option", required=False, nargs=2, type=click.Tuple([unicode, unicode]), default=(None, None))
def tool(n_jobs, is_testing, column, mode, week, is_output, option):
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
    elif mode == "cache":
        cache_median(TRAIN, column, week)
    elif mode == "solution":
        solution, column = option

        if solution == "ftlr":
            folder = os.path.join(SPLIT_PATH, COLUMNS[column], "test")
            submission_folder = os.path.join(FTLR_SOLUTION_PATH, "train" if week < 10 else "test", "week={}".format(week), COLUMNS[column])
            create_folder("{}/1.txt".format(submission_folder))

            Parallel(n_jobs=n_jobs)(delayed(ftlr_solution)(folder, os.path.basename(filepath).replace(".csv", ""), submission_folder) for filepath in glob.iglob(os.path.join(folder, "*.csv")))
        elif solution == "median":
            groups = None
            if MONGODB_COLUMNS[COLUMN_ROUTE] == column:
                groups = ROUTE_GROUPS
            elif MONGODB_COLUMNS[COLUMN_AGENCY] == column:
                groups = AGENCY_GROUPS
            else:
                raise NotImplementError

            solution = (load_median_solution(week-1, column, groups), groups)

            if week < 10:
                filepaths = os.path.join(SPLIT_PATH, COLUMNS[column], "train", "*.csv")
                for filepath in glob.iglob(filepaths):
                    fileid = os.path.basename(filepath).replace(".csv", "")
                    output_filepath = os.path.join(MEDIAN_SOLUTION_PATH, "train", "week={}".format(week), COLUMNS[column], "submission_{}.csv".format(fileid))

                    median_solution(week, output_filepath, filepath, solution)
            else:
                filepaths = os.path.join(SPLIT_PATH, COLUMNS[column], "test", "*.csv")
                for filepath in glob.iglob(filepaths):
                    fileid = os.path.basename(filepath).replace(".csv", "")
                    output_filepath = os.path.join(MEDIAN_SOLUTION_PATH, "test", "week={}".format(week), COLUMNS[column], "submission_{}.csv".format(fileid))

                    if not os.path.exists(output_filepath):
                        median_solution(week, output_filepath, filepath, solution)
        elif solution == "cc":
            if week < 10:
                folder = os.path.join(SPLIT_PATH, COLUMNS[column], "train")
                Parallel(n_jobs=n_jobs)(delayed(cc)(column, int(os.path.basename(filepath).replace(".csv", "")), week, is_output) for filepath in glob.iglob(os.path.join(folder, "*.csv")))
            else:
                pass
        elif solution == "regression":
            week_y = 10
            solutions=["ftlr", "median"]
            nfold=3
            files = os.path.join(SPLIT_PATH, COLUMNS[column], "test", "*.csv")

            Parallel(n_jobs=n_jobs)(delayed(regression_solution)((column, int(os.path.basename(filepath).replace(".csv", ""))), week, week_y, solutions, nfold) for filepath in glob.iglob(os.path.join(files)))
    elif mode == "ensemble":
        filepaths, output_filepath = option

        ensemble_solution(filepaths.split(","), output_filepath)
    else:
        log("Not found this mode {}".format(mode), ERROR)
        sys.exit(101)

if __name__ ==  "__main__":
    tool()

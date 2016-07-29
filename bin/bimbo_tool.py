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
from bimbo.constants import SPLIT_PATH, STATS_PATH, FTLR_SOLUTION_PATH, MEDIAN_SOLUTION_CACHE, MEDIAN_SOLUTION_PATH, CC_SOLUTION_PATH
from bimbo.constants import ROUTE_GROUPS, AGENCY_GROUPS
from bimbo.constants import TRAIN_FILE, TEST_FILE, TESTING_TRAIN_FILE, TESTING_TEST_FILE
from bimbo.solutions import ensemble_solution, median_solution, ftlr_solution, regression_solution, cc_solution, cache_median
from bimbo.tools import purge_duplicated_records, hierarchical_folder_structure, repair_missing_records, aggregation

TRAIN = TRAIN_FILE
TEST = TEST_FILE

@click.command()
@click.option("--n-jobs", default=1, help="number of thread")
@click.option("--is-testing", is_flag=True, help="testing mode")
@click.option("--column", default=None, help="agency_id|channel_id|route_id|client_id|product_id")
@click.option("--mode", required=True, help="purge|restructure")
@click.option("--week", default=9, help="week number(4-9)")
@click.option("--option", required=False, nargs=2, type=click.Tuple([unicode, unicode]), default=(None, None))
def tool(n_jobs, is_testing, column, mode, week, option):
    global TRAIN, TEST

    if is_testing:
        TRAIN = TESTING_TRAIN_FILE
        TEST = TESTING_TEST_FILE

    if mode == "purge":
        solution_type, column = option

        purge_duplicated_records(week, solution_type, column)
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
        solution, column_name = option
        cs = "lag1_client_product,lag1_median_channel,lag1_median_column,lag2_client_product,lag2_median_channel,lag2_median_column,lag3_client_product,lag3_median_channel,lag3_median_column,return_1,return_2,return_3,trend_1,trend_2"
        cs = "return_1,return_2"

        if solution == "ftlr":
            folder = os.path.join("{}.category".format(SPLIT_PATH), COLUMNS[column_name], "test")
            submission_folder = os.path.join(FTLR_SOLUTION_PATH, "train" if week < 10 else "test", "week={}".format(week), COLUMNS[column_name])
            create_folder("{}/1.txt".format(submission_folder))

            columns = [COLUMNS[c] for c in column.split(",")]
            columns.extend(cs.split(","))
            #columns.extend(["median_route_solution", "median_agency_solution"])
            #columns.remove(COLUMNS[column_name])

            log("Use {} to be the attributes".format(columns), INFO)

            Parallel(n_jobs=n_jobs)(delayed(ftlr_solution)(folder, os.path.basename(filepath).replace(".csv", ""), submission_folder, week, ",".join(columns)) for filepath in glob.iglob(os.path.join(folder, "*.csv")))
        elif solution == "median":
            groups = None
            if MONGODB_COLUMNS[COLUMN_ROUTE] == column_name:
                groups = ROUTE_GROUPS
            elif MONGODB_COLUMNS[COLUMN_AGENCY] == column_name:
                groups = AGENCY_GROUPS
            else:
                raise NotImplementedError

            solution = (load_median_solution(week-1, column_name, groups), groups)

            if week < 10:
                filepaths = os.path.join(SPLIT_PATH, COLUMNS[column_name], "train", "*.csv")
                for filepath in glob.iglob(filepaths):
                    fileid = os.path.basename(filepath).replace(".csv", "")
                    output_filepath = os.path.join(MEDIAN_SOLUTION_PATH, "train", "week={}".format(week), COLUMNS[column_name], "submission_{}.csv".format(fileid))

                    median_solution(week, output_filepath, filepath, solution)
            else:
                filepaths = os.path.join(SPLIT_PATH, COLUMNS[column_name], "test", "*.csv")
                for filepath in glob.iglob(filepaths):
                    fileid = os.path.basename(filepath).replace(".csv", "")
                    output_filepath = os.path.join(MEDIAN_SOLUTION_PATH, "test", "week={}".format(week), COLUMNS[column_name], "submission_{}.csv".format(fileid))

                    if not os.path.exists(output_filepath):
                        median_solution(week, output_filepath, filepath, solution)
        elif solution == "cc":
            if week < 10:
                folder = os.path.join(SPLIT_PATH, COLUMNS[column_name], "train")
                cc_solution_folder = os.path.join(CC_SOLUTION_PATH, "train", "week={}".format(week), COLUMNS[column_name])
                create_folder("{}/1.txt".format(cc_solution_folder))

                f = lambda x: os.path.join(cc_solution_folder, os.path.basename(x))

                for filepath in glob.iglob(os.path.join(folder, "*.csv")):
                    log(filepath)
                    cc_solution(week, "cc", column_name, filepath, f(filepath))

                Parallel(n_jobs=n_jobs)(delayed(cc_solution)(week, "cc", column_name, filepath, f(filepath)) for filepath in glob.iglob(os.path.join(folder, "*.csv")))
            else:
                folder = os.path.join(SPLIT_PATH, COLUMNS[column_name], "test")
                cc_solution_folder = os.path.join(CC_SOLUTION_PATH, "test", "week={}".format(week), COLUMNS[column_name])
                create_folder("{}/1.txt".format(cc_solution_folder))

                f = lambda x: os.path.join(cc_solution_folder, os.path.basename(x))

                Parallel(n_jobs=n_jobs)(delayed(cc_solution)(week, "cc", column_name, filepath, f(filepath)) for filepath in glob.iglob(os.path.join(folder, "*.csv")))
        elif solution == "regression":
            week_y = 10
            solutions=["ftlr", "median"]
            nfold=3
            files = os.path.join(SPLIT_PATH, COLUMNS[column_name], "test", "*.csv")

            f = lambda x: int(os.path.basename(x).replace(".csv", ""))
            Parallel(n_jobs=n_jobs)(delayed(regression_solution)((column_name, f(filepath)), week, week_y, solutions, nfold) for filepath in glob.iglob(os.path.join(files)))
    elif mode == "ensemble":
        filepaths, output_filepath = option

        ensemble_solution(filepaths.split(","), output_filepath)
    else:
        log("Not found this mode {}".format(mode), ERROR)
        sys.exit(101)

if __name__ ==  "__main__":
    tool()

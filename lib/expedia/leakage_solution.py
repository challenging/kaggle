#!/usr/bin/env python

import os
import sys

import glob
import re
import datetime

import threading
import Queue

from utils import log, INFO
from leakage import prepare_arrays_match, gen_submission

def worker(working_queue, folder_submission):
    while True:
        filepath_testing = working_queue.get()

        filename = os.path.basename(filepath_testing)
        filepath_training, filepath_submission = filepath_testing.replace("test", "train"), os.path.join(folder_submission, filename.replace("test", "submission"))

        best_hotels_search_dest, best_hotels_search_dest1, best_hotels_user_location, best_hotels_od_ulc, best_hotels_country, popular_hotel_cluster = prepare_arrays_match(filepath_training)
        with open(filepath_submission, "wb") as OUTPUT:
            for user_id, filled in gen_submission(filepath_testing, best_hotels_search_dest, best_hotels_search_dest1, best_hotels_user_location, best_hotels_od_ulc, best_hotels_country, popular_hotel_cluster):
                OUTPUT.write("{},{}\n".format(user_id, " ".join(filled)))

            log("Create the sub-submission file in {}".format(filepath_submission), INFO)

        working_queue.task_done()

def main(f_pattern, n_threads=4):
    working_queue = Queue.Queue()

    folder = os.path.dirname(f_pattern)
    folder_name = os.path.basename(folder)

    method = None
    for filepath_testing in glob.iglob(f_pattern):
        if method == None:
            methods = re.search("test_([\w_|]*)=[\d|]+.csv", filepath_testing)
            if methods:
                method = methods.groups()[0]

        working_queue.put(filepath_testing)

    if method == None:
        method = "all"

    # Create submission folder
    folder_submission = folder.replace("test", "submission_method={}_plus=hotelcountry|userlocation|weekday|book_year_{}".format(method, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")))
    if not os.path.isdir(folder_submission):
        os.makedirs(folder_submission)

    for idx in range(n_threads):
        thread = threading.Thread(target=worker, kwargs={"working_queue": working_queue,
                                                         "folder_submission": folder_submission})
        thread.setDaemon(True)
        thread.start()

    working_queue.join()

    filepath_submission = "{}/final_submission.csv".format(folder_submission)
    with open(filepath_submission, "wb") as OUTPUT:
        OUTPUT.write("id,hotel_cluster\n")

        total_count = 0
        for filepath in glob.iglob("{}/*csv".format(folder_submission)):
            a = open(filepath, "rb")
            lines = a.readlines()
            a.close()

            total_count += len(lines)

            OUTPUT.writelines(lines)

    log("Create the final submission file in {}".format(filepath_submission), INFO)

if __name__ == "__main__":
    filepath_train = "/Users/RungChiChen/Documents/programs/kaggle/cases/Expedia Hotel Recommendations/input/train/train.csv"
    filepath_test = "/Users/RungChiChen/Documents/programs/kaggle/cases/Expedia Hotel Recommendations/input/test/test.csv"

    f_patterns = [
                  "/Users/RungChiChen/Documents/programs/kaggle/cases/Expedia Hotel Recommendations/input/test/test.csv",
                  #"/Users/RungChiChen/Documents/programs/kaggle/cases/Expedia Hotel Recommendations/input/test/user_location_country/test_user_location_country=*.csv",
                  #"/Users/RungChiChen/Documents/programs/kaggle/cases/Expedia Hotel Recommendations/input/test/site_name/test_site_name=*.csv",
                  #"/Users/RungChiChen/Documents/programs/kaggle/cases/Expedia Hotel Recommendations/input/test/date_time/*.csv"
                 ]

    for pattern in f_patterns:
        main(pattern, 6)

import os
import sys

import datetime

import md5
import copy
import logging
import subprocess

CRITICAL = logging.CRITICAL
ERROR = logging.ERROR
WARN = logging.WARNING
INFO = logging.INFO
DEBUG = logging.DEBUG

# Log Level: CRITICAL   > ERROR   >  WARNING   >  INFO   > DEBUG   > NOTSET

# Produce formater first
formatter = logging.Formatter('%(asctime)s - [%(name)s] - [%(threadName)s] - [%(levelname)s] - %(message)s')

# Setup Handler
console = logging.StreamHandler()
console.setLevel(INFO)
console.setFormatter(formatter)

# Setup Logger
logger = logging.getLogger("Kaggle")
logger.addHandler(console)
logger.setLevel(INFO)

def log(msg, level=logging.INFO):
    '''
    Logging Function
    ==========================
    :param msg|string: the logging message
    :param level|int(defined by built-in module, logging): the logging level
    :return: None
    '''

    if level == logging.CRITICAL:
        logger.critical(msg)
    elif level == logging.ERROR:
        logger.error(msg)
    elif level == logging.WARNING:
        logger.warn(msg)
    elif level == logging.INFO:
        logger.info(msg)
    elif level == logging.DEBUG:
        logger.debug(msg)
    else:
        logger.notset(msg)

def make_a_stamp(model_setting):
    m = md5.new()
    m.update(str(model_setting))

    return m.hexdigest()

def create_folder(filepath):
    folder = os.path.dirname(filepath)

    if not os.path.isdir(folder):
        try:
            os.makedirs(folder)
            log("Create folder in {}".format(folder), INFO)
        except OSError as e:
            if str(e).find("File Exists") == -1:
                log(e)

def split_file_by_size(filepath, n_size=32*1024*1024):
    part_no = 1
    chunk = []
    with open(filepath, "rb") as INPUT:
        for line in INPUT:
            chunk.append(line)

            if sys.getsizeof(chunk) > n_size:
                filepath_part = "{}.part.{}".format(filepath, part_no)
                with open(filepath_part, "wb") as OUTPUT:
                    OUTPUT.writelines(chunk)

                log("write sub-file in {}".format(filepath_part), INFO)

                part_no += 1
                chunk = []

    filepath_part = "{}.part.{}".format(filepath, part_no)
    with open(filepath_part, "wb") as OUTPUT:
        OUTPUT.writelines(chunk)

    log("write sub-file in {}".format(filepath_part), INFO)

def simple_split(arr, idx):
    return arr[idx]

def weekday_split(arr, idx):
    weekday = -1
    parsered_date = arr[idx]

    try:
        if parsered_date.find(":") == -1:
            weekday = datetime.datetime.strptime(parsered_date, "%Y-%m-%d").weekday()
        else:
            weekday = datetime.datetime.strptime(parsered_date, "%Y-%m-%d %H:%M:%S").weekday()
    except ValueError as e:
        pass

    return str(weekday)

def split_file_by_column(filepath, idxs):
    combine = lambda x: "|".join(x)

    headers, column_name = None, []
    folder = os.path.dirname(filepath)
    filename = os.path.basename(filepath)

    outputs = {}

    with open(filepath, "rb") as INPUT:
        for line in INPUT:
            arr = line.strip().split(",")

            if not column_name:
                for idx, method in idxs:
                    column_name.append(arr[idx])

                folder += "/{}".format(combine(column_name))
                if not os.path.isdir(folder):
                    os.makedirs(folder)

                log("Save files in {}".format(folder))

                headers = line
            else:
                value = []
                for idx, method in idxs:
                    value.append(method(arr, idx))

                key = combine(value)
                outputs.setdefault(key, [])
                outputs[key].append(line)

    for value, lines in outputs.items():
        with open("{}/{}_{}={}.csv".format(folder, filename.replace(".csv", ""), combine(column_name), value), "wb") as OUTPUT:
            lines.insert(0, headers)
            OUTPUT.writelines(lines)

        log("There are {} records in {}".format(len(lines), value))

if __name__ == "__main__":
    #split_file_by_column("/Users/RungChiChen/Documents/programs/kaggle/cases/Expedia Hotel Recommendations/input/train/train.csv", [(0, weekday_split)])#[(3, simple_split), (11, weekday_split)])
    #split_file_by_column("/Users/RungChiChen/Documents/programs/kaggle/cases/Expedia Hotel Recommendations/input/test/test.csv", [(1, weekday_split)])#[(4, simple_split), (12, weekday_split)])

    pass

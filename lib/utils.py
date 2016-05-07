import os
import sys

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
        os.makedirs(folder)
        log("Create folder in {}".format(folder), INFO)

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

def split_file_by_column(filepath, idx, split_char=","):
    column_name = None
    folder = os.path.dirname(filepath)
    filename = os.path.basename(filepath)

    outputs = {}

    with open(filepath, "rb") as INPUT:
        for line in INPUT:
            arrs = line.split(split_char)
            if column_name == None:
                column_name = arrs[idx]
            else:
                value = arrs[idx]

                outputs.setdefault(value, open("{}/{}_site_name={}.csv".format(folder, filename, value), "wb"))
                outputs[value].write(line)

    for output in outputs.values():
        output.close()

if __name__ == "__main__":
    #split_file("/Users/rongqichen/Documents/programs/kaggle/cases/Expedia Hotel Recommendations/input/train/train.csv")
    split_file_by_column("/Users/rongqichen/Documents/programs/kaggle/cases/Expedia Hotel Recommendations/input/train/train.csv", 1)

import logging
import os
import copy
import md5
import collections

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

    #m.update(str(collections.OrderedDict(sorted(model_setting.items()))))

    return m.hexdigest()

def create_folder(filepath):
    folder = os.path.dirname(filepath)

    if not os.path.isdir(folder):
        os.makedirs(folder)
        log("Create folder in {}".format(folder), INFO)

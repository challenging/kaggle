#!/usr/bin/env python

import os
import json
import zlib
import time

import pandas as pd
import numpy as np

import pymongo
import beanstalkc

from utils import log
from utils import DEBUG, INFO, WARN
from bimbo.constants import get_mongo_connection, get_cc_mongo_collection, get_prediction_mongo_collection, get_median
from bimbo.constants import COMPETITION_CC_NAME, IP_BEANSTALK, PORT_BEANSTALK, TIMEOUT_BEANSTALK
from bimbo.constants import MONGODB_PREDICTION_COLLECTION, MONGODB_STATS_CC_COLLECTION, MONGODB_PREDICTION_DATABASE, MONGODB_CC_DATABASE
from bimbo.constants import COLUMN_ROW, COLUMN_WEEK, COLUMN_AGENCY, COLUMN_CHANNEL, COLUMN_ROUTE, COLUMN_PRODUCT, COLUMN_CLIENT, COLUMNS, MONGODB_COLUMNS
from bimbo.constants import SPLIT_PATH, NON_PREDICTABLE, TOTAL_WEEK

def adjust_prediction_cc(values, prediction_cc):
    if values[end_idx-1] == 0 and np.sum(np.array(values[0:end_idx]) == 0) > 4:
        prediction_cc = 0
    elif prediction_cc > 0:
        prediction_cc = prediction_cc*0.71 + 0.243

    return prediction_cc

def cc_calculation(week, filetype, product_id, predicted_rows, history, threshold_value=0, progress_prefix=None):
    global TOTAL_WEEK

    end_idx = (TOTAL_WEEK-3) - (TOTAL_WEEK-week)

    count = 0
    for client_id in predicted_rows.keys():
        values = history[client_id]

        record = {"groupby": MONGODB_COLUMNS[filetype[0]], MONGODB_COLUMNS[filetype[0]]: int(filetype[1]), "product_id": product_id, "client_id": int(client_id), "history": values, "cc": []}
        prediction = {"row_id": predicted_rows[client_id],
                      "history": values,
                      "prediction_cc": NON_PREDICTABLE}

        if np.sum(values[0:end_idx]) == 0:
            yield record, prediction

            continue

        client_mean = np.mean(values[1:end_idx])
        cc_client_ids, cc_matrix = [client_id], [values[1:end_idx]]
        for cc_client_id, cc_client_values in history.items():
            if client_id != cc_client_id:
                cc_client_ids.append(cc_client_id)
                cc_matrix.append(cc_client_values[0:end_idx-1])

        results_cc = {}
        if len(cc_client_ids) > 1:
            cc_results = np.corrcoef(cc_matrix)[0]
            results_cc = dict(zip(cc_client_ids, cc_results))

            matrix = []
            num_sum, num_count = 0, 0.00000001
            for c, value in results_cc.items():
                if c == client_id or np.isnan(value):
                    continue
                elif np.sum(np.array(history[c][:end_idx-1]) == 0) < 3:
                    diff = history[c][end_idx-1] - history[c][end_idx-2]
                    if diff != 0:
                        ratio = client_mean/np.mean(history[c][0:end_idx-1])
                        score = diff*value*ratio

                        num_sum += score
                        num_count += 1

                        matrix.append({client_id: int(c), "value": value})
                        log("!!!! {} - {} >>> {}, {}-{}={}, {}".format(history[client_id], history[c], value, history[c][end_idx-1], history[c][end_idx-2], score, num_sum/num_count), DEBUG)

            prediction_cc = values[end_idx-1] + (num_sum/num_count)

            record["cc"] = matrix
            prediction["prediction_cc"] = prediction_cc
        else:
            log("Found only {} in {}({})".format(client_id, product_id, len(history)), WARN)

        count += 1

        log("{} {}/{} >>> {} - {} - {} - {:4f}".format(\
            "{}/{}".format(progress_prefix[0], progress_prefix[1]) if progress_prefix else "", count, len(predicted_rows), product_id, client_id, history[client_id], prediction["prediction_cc"]), DEBUG)

        yield record, prediction

def median_calculation(week, filetype, product_id, predicted_rows, median_solution, progress_prefix=None):
    global TOTAL_WEEK

    count = 0
    for client_id in predicted_rows.keys():
        prediction_median = get_median(median_solution[0], median_solution[1], {filetype[0]: filetype[1], COLUMN_PRODUCT: product_id, COLUMN_CLIENT: client_id})

        prediction = {"row_id": predicted_rows[client_id],
                      "prediction_median": prediction_median}

        count += 1

        log("{} {}/{} >>> {} - {} - {:4f}".format(\
            "{}/{}".format(progress_prefix[0], progress_prefix[1]) if progress_prefix else "", count, len(predicted_rows), product_id, client_id, prediction["prediction_median"]), DEBUG)

        yield prediction

def cc_consumer(column, task=COMPETITION_CC_NAME):
    client = get_mongo_connection()

    talk = beanstalkc.Connection(host=IP_BEANSTALK, port=PORT_BEANSTALK)
    talk.watch(task)

    while True:
        try:
            job = talk.reserve(timeout=TIMEOUT_BEANSTALK)
            if job:
                o = json.loads(zlib.decompress(job.body))
                week, filetype = o[COLUMN_WEEK], o["filetype"]
                predicted_rows = o["predicted_rows"]
                product_id, history = o["product_id"], o["history"]

                mongodb_cc_database, mongodb_cc_collection = o["mongodb_cc"]
                cc_collection = client[mongodb_cc_database][mongodb_cc_collection]

                mongodb_prediction_database, mongodb_prediction_collection = o["mongodb_prediction"]
                prediction_collection = client[mongodb_prediction_database][mongodb_prediction_collection]

                for client_id, row_ids in predicted_rows.copy().items():
                    for row_id in row_ids:
                        c = prediction_collection.count({COLUMN_ROW: row_id})
                        if c > 0:
                            predicted_rows.remove(row_id)

                    if len(predicted_rows[client_id]) == 0:
                        del predicted_rows[client_id]
                        log("Delete the {} from the predicted_rows".format(client_id), INFO)

                if "version" not in o or o["version"] < 1.1:
                    log("Skip this {} of {} because of the lower version".format(product_id, filetype), INFO)
                else:
                    timestamp_start = time.time()

                    records, predictions = [], []
                    log("There are {}/{} predicted_rows/history in {} of {}".format(len(predicted_rows), len(history), product_id, filetype), INFO)
                    for cc, prediction in cc_calculation(week, (COLUMNS[filetype[0]], filetype[1]), product_id, predicted_rows, history):
                        #records.append(cc)

                        for row_id in prediction["row_id"]:
                            predictions.append({"row_id": row_id, "history": prediction["history"], "prediction": prediction["prediction_cc"]})

                    # The storage is not enought to store matrix of CC
                    #if records:
                    #    cc_collection.insert_many(records)

                    if predictions:
                        prediction_collection.insert_many(predictions)

                    timestamp_end = time.time()
                    log("Cost {:4f} secends to insert {}/{} records into mongodb".format(timestamp_end-timestamp_start, len(records), len(predictions)), INFO)

                job.delete()
        except beanstalkc.BeanstalkcException as e:
            log("Error occurs, {}".format(e), WARN)
        except KeyError as e:
            o = json.loads(zlib.decompress(job.body))
            if "ending" in o:
                job.delete()
            else:
                log("{} - {}".format(e, o), WARN)
        except Exception as e:
            raise

    talk.close()
    client.close()

def median_consumer(median_solution, task=COMPETITION_CC_NAME):
    client = get_mongo_connection()

    mongodb_prediction_database, mongodb_prediction_collection = MONGODB_PREDICTION_DATABASE, get_prediction_mongo_collection("median_{}".format(MONGODB_COLUMNS[COLUMN_PRODUCT]))
    prediction_collection = client[mongodb_prediction_database][mongodb_prediction_collection]

    log("Ready to use {}".format(task))
    talk = beanstalkc.Connection(host=IP_BEANSTALK, port=PORT_BEANSTALK)
    talk.watch(task)

    while True:
        try:
            job = talk.reserve(timeout=TIMEOUT_BEANSTALK)
            if job:
                o = json.loads(zlib.decompress(job.body))
                week, filetype = o[COLUMN_WEEK], o["filetype"]
                predicted_rows = o["predicted_rows"]

                product_id, history = o["product_id"], o["history"]

                predictions = []
                for prediction in median_calculation(week, (COLUMNS[filetype[0]], filetype[1]), product_id, predicted_rows, median_solution):
                    for row_id in prediction["row_id"]:
                        predictions.append({"row_id": row_id, "prediction": prediction["prediction_median"]})

                prediction_collection.insert_many(predictions)
                log("Insert {} records into mongodb from {} of {}".format(len(predictions), product_id, filetype), INFO)

                # To avoid BulkWriteErrors if the speed is too fast
                time.sleep(0.05)

                job.delete()
        except beanstalkc.BeanstalkcException as e:
            log("Error occurs, {}".format(e), WARN)
        except KeyError as e:
            o = json.loads(zlib.decompress(job.body))
            if "ending" in o:
                job.delete()
            else:
                log("{} - {}".format(e, o), WARN)
        except Exception as e:
            raise

    talk.close()
    client.close()

def get_history(filepath_train, filepath_test, shift_week=3, week=[3, TOTAL_WEEK], collection=None):
    history = {}

    with open(filepath_train, "rb") as INPUT:
        header = True

        for line in INPUT:
            if header:
                header = False
                continue

            w, agency_id, channel_id, route_id, client_id, product_id, sales_unit, sales_price, return_unit, return_price, prediction_unit = line.strip().split(",")

            w = int(w)
            client_id = int(client_id)
            product_id = int(product_id)
            prediction_unit = int(prediction_unit)

            history.setdefault(product_id, {})
            history[product_id].setdefault(client_id, [0 for _ in range(week[0], week[1])])
            history[product_id][client_id][w-shift_week] = np.log1p(prediction_unit)

    others = {}
    predicted_rows = {}
    with open(filepath_test, "rb") as INPUT:
        header = True

        for line in INPUT:
            if header:
                header = False
                continue

            if filepath_train == filepath_test:
                w, agency_id, channel_id, route_id, client_id, product_id, sales_unit, sales_price, return_unit, return_price, prediction_unit = line.strip().split(",")
                row_id = "T{}".format(int(time.time()))
            else:
                row_id, w, agency_id, channel_id, route_id, client_id, product_id = line.strip().split(",")
                row_id = int(row_id)

            w = int(w)
            client_id = int(client_id)
            product_id = int(product_id)

            history.setdefault(product_id, {})
            history[product_id].setdefault(client_id, [0 for _ in range(week[0], week[1])])

            # workaround
            #count = collection.count({"row_id": row_id})
            #if count == 0:
            #    log("Miss Row ID: {}".format(row_id))
            #else:
            #    continue

            predicted_rows.setdefault(product_id, {})
            predicted_rows[product_id].setdefault(client_id, [])
            predicted_rows[product_id][client_id].append(row_id)

            key = "{}_{}".format(client_id, product_id)
            others.setdefault(key, [])
            others[key].append(",".join([str(w), agency_id, channel_id, route_id, str(client_id), str(product_id)]))

    return history, predicted_rows, others

def producer(week, filetype, solution_type, task=COMPETITION_CC_NAME, ttr=TIMEOUT_BEANSTALK):
    filepath_train = os.path.join(SPLIT_PATH, COLUMNS[filetype[0]], "train", "{}.csv".format(filetype[1]))
    filepath_test = os.path.join(SPLIT_PATH, COLUMNS[filetype[0]], "test", "{}.csv".format(filetype[1]))

    if os.path.exists(filepath_test):
        talk = beanstalkc.Connection(host=IP_BEANSTALK, port=PORT_BEANSTALK)
        talk.use(task)

        client = get_mongo_connection()

        mongodb_cc_database, mongodb_cc_collection = MONGODB_CC_DATABASE, get_cc_mongo_collection(MONGODB_COLUMNS[COLUMN_PRODUCT])
        client[mongodb_cc_database][mongodb_cc_collection].create_index([("groupby", pymongo.ASCENDING), (filetype[0], pymongo.ASCENDING), ("client_id", pymongo.ASCENDING), ("product_id", pymongo.ASCENDING)])

        mongodb_prediction_database, mongodb_prediction_collection = MONGODB_PREDICTION_DATABASE, get_prediction_mongo_collection("{}_{}_product_client_log1p".format(solution_type, filetype[0]))
        client[mongodb_prediction_database][mongodb_prediction_collection].create_index("row_id")

        history, predicted_rows, _ = get_history(filepath_train, filepath_test, collection=client[mongodb_prediction_database][mongodb_prediction_collection])
        for product_id, info in predicted_rows.items():
            request = {"version": 1.1,
                       "product_id": product_id,
                       COLUMN_WEEK: week,
                       "filetype": list(filetype),
                       "history": history[product_id],
                       "predicted_rows": predicted_rows[product_id],
                       "mongodb_cc": (mongodb_cc_database, mongodb_cc_collection),
                       "mongodb_prediction": (mongodb_prediction_database, mongodb_prediction_collection)}

            talk.put(zlib.compress(json.dumps(request)), ttr=TIMEOUT_BEANSTALK)
            log("Put request(product_id={} from {}) into the {}".format(product_id, os.path.basename(filepath_train), task), INFO)

        client.close()
        talk.close()
    else:
        log("Not found {} so skipping it".format(filepath_test), INFO)

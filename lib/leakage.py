#!/usr/bin/env python

import os
import sys
import time
import datetime

import re
import glob
import collections
import threading
import Queue

from heapq import nlargest
from operator import itemgetter

from load import save_cache, load_cache
from utils import log, INFO, WARN

SPLIT_CHAR = ","
UNKNOWN_HOTEL_CLUSTER = "X"

def prepare_arrays_match(filepath):
    best_hotels_od_ulc = {}
    best_hotels_search_dest, best_hotels_search_dest_formula = {}, lambda x: 3 + 17*x
    best_hotels_user_location, best_hotels_user_location_formula = {}, lambda x: 3 + 17*x
    best_hotels_search_dest1, best_hotels_search_dest1_formula = {}, lambda x: 3 + 17*x
    best_hotels_country, best_hotels_country_formula = {}, lambda x: 1 + 5*x
    popular_hotel_cluster = {}

    def cluster_calculation(key, hotel, d, v):
        d.setdefault(key, {})
        d[key].setdefault(hotel_cluster, 0)
        d[key][hotel_cluster] += v

    # Calc counts
    if os.path.exists(filepath):
        with open(filepath, "rb") as INPUT:
            for line in INPUT:
                line = line.strip()

                arr = line.split(",")
                user_city = arr[5]

                if not user_city.isdigit():
                    continue

                weekday = str(datetime.datetime.strptime(arr[0], "%Y-%m-%d %H:%M:%S").weekday())
                book_year = int(arr[0][:4])
                user_country = arr[3]
                user_region = arr[4]
                orig_destination_distance = arr[6]
                srch_destination_id = arr[16]
                is_booking = int(arr[18])
                hotel_country = arr[21]
                hotel_market = arr[22]
                hotel_cluster = arr[23]

                if user_city != '' and orig_destination_distance != '':
                    key = (user_city, orig_destination_distance)
                    cluster_calculation(key, hotel_cluster, best_hotels_od_ulc, 1)

                if srch_destination_id != '' and hotel_country != '' and hotel_market != '' and book_year == 2014:
                    key = (srch_destination_id, hotel_country, hotel_market)
                    cluster_calculation(key, hotel_cluster, best_hotels_search_dest, best_hotels_search_dest_formula(is_booking))

                if srch_destination_id != "":
                    key = (srch_destination_id)
                    #cluster_calculation(key, hotel_cluster, best_hotels_user_location, best_hotels_user_location_formula(is_booking))
                    cluster_calculation(key, hotel_cluster, best_hotels_search_dest1, best_hotels_search_dest1_formula(is_booking))

                if user_city != "" and srch_destination_id != "":
                    key = (user_city, srch_destination_id)
                    #cluster_calculation(key, hotel_cluster, best_hotels_search_dest1, best_hotels_search_dest1_formula(is_booking))
                    cluster_calculation(key, hotel_cluster, best_hotels_user_location, best_hotels_user_location_formula(is_booking))

                if hotel_country != "":
                    key = (weekday, hotel_country)
                    cluster_calculation(key, hotel_cluster, best_hotels_country, best_hotels_country_formula(is_booking))

                key = (weekday)
                cluster_calculation(key, hotel_cluster, popular_hotel_cluster, 1)
    else:
        log("Not found {}".format(filepath), WARN)

    return best_hotels_search_dest, best_hotels_search_dest1, best_hotels_user_location, best_hotels_od_ulc, best_hotels_country, popular_hotel_cluster

def gen_submission(filepath_testing, best_hotels_search_dest, best_hotels_search_dest1, best_hotels_user_location, best_hotels_od_ulc, best_hotels_country, popular_hotel_cluster):
    topclasters = nlargest(5, sorted(popular_hotel_cluster.items()), key=itemgetter(1))

    def fill(filled, d):
        topitems = nlargest(5, d.items(), key=itemgetter(1))
        for i in range(len(topitems)):
            if topitems[i][0] in filled:
                continue
            if len(filled) == 5:
                break

            filled.append(topitems[i][0])

    filepath_testing_profile = filepath_testing + "_profile.csv"

    count_s1, count_s2, count_s3, count_s4, count_s5, count_popular = 0, 0, 0, 0, 0, 0
    with open(filepath_testing, "rb") as INPUT:
        for line in INPUT:
            line = line.strip()

            arr = line.split(",")
            user_id = arr[0]
            if not user_id.isdigit():
                continue

            weekday = str(datetime.datetime.strptime(arr[1], "%Y-%m-%d %H:%M:%S").weekday())
            user_location_country = arr[4]
            user_location_region = arr[5]
            user_location_city = arr[6]
            orig_destination_distance = arr[7]
            srch_destination_id = arr[17]
            hotel_country = arr[20]
            hotel_market = arr[21]

            filled = []

            s1 = (user_location_city, orig_destination_distance)
            if s1 in best_hotels_od_ulc:
                d = best_hotels_od_ulc[s1]
                fill(filled, d)

                count_s1 += 1

            s2 = (srch_destination_id, hotel_country, hotel_market)
            if s2 in best_hotels_search_dest:
                d = best_hotels_search_dest[s2]
                fill(filled, d)

                count_s2 += 1

            s3 = (srch_destination_id)
            if s3 in best_hotels_user_location:
                d = best_hotels_user_location[s3]
                fill(filled, d)

                count_s3 += 1

            s4 = (user_location_city, srch_destination_id)
            if s4 in best_hotels_search_dest1:
                d = best_hotels_search_dest1[s4]
                fill(filled, d)

                count_s4 += 1

            s5 = (weekday, hotel_country)
            if s5 in best_hotels_country:
                d = best_hotels_country[s5]
                fill(filled, d)

                count_s5 += 1

            if len(filled) < 5:
                count_popular += 1

            for i in range(len(topclasters)):
                if topclasters[i][0] in filled:
                    continue
                if len(filled) == 5:
                    break

                filled.append(topclasters[i][0])

            yield (user_id, filled)

    log("There are {} / {} / {} / {} / {} / {} matching records".format(count_s1, count_s2, count_s3, count_s4, count_s5, count_popular), INFO)

def global_solution(filepath):
    filepath_pkl = "{}.pkl".format(filepath)

    best_hotels_search_dest, best_hotels_od_ulc, popular_hotel_cluster = {}, {}, {}
    if os.path.exists(filepath_pkl):
        best_hotels_search_dest, best_hotels_od_ulc, popular_hotel_cluster = load_cache(filepath_pkl)
    else:
        best_hotels_search_dest, best_hotels_od_ulc, popular_hotel_cluster = prepare_arrays_match(filepath)
        save_cache((best_hotels_search_dest, best_hotels_od_ulc, popular_hotel_cluster), filepath_pkl)

    return best_hotels_search_dest, best_hotels_od_ulc, popular_hotel_cluster

def booking_chain(filepath_train, filepath_test, filepath_chain, datetime_idx=0, base_idx=7, booking_idx=18, target_idx=16):
    global SPLIT_CHAR, UNKNOWN_HOTEL_CLUSTER

    pure_cluster_chain = {}

    filepath_cluster_pkl = filepath_chain
    if os.path.exists(filepath_cluster_pkl):
        pure_cluster_chain = load_cache(filepath_cluster_pkl)

        return pure_cluster_chain

    booking_cluster_chain = {}
    with open(filepath_train, "rb") as INPUT:
        for line in INPUT:
            arr = line.strip().split(SPLIT_CHAR)
            date, user_id, target, is_booking = arr[datetime_idx], arr[base_idx], arr[target_idx], arr[booking_idx]
            if not user_id.isdigit():
                continue

            booking_cluster_chain.setdefault(user_id, {})
            booking_cluster_chain[user_id][date] = "{}({})".format(target, is_booking)

    with open(filepath_test, "rb") as INPUT:
        for line in INPUT:
            arr = line.strip().split(SPLIT_CHAR)
            date, user_id = arr[datetime_idx+1], arr[base_idx+1]
            if not user_id.isdigit():
                continue

            target = UNKNOWN_HOTEL_CLUSTER
            if target_idx != -1:
                target = arr[target_idx+1]

            booking_cluster_chain.setdefault(user_id, {})
            booking_cluster_chain[user_id][date] = target

    with open(filepath_cluster_pkl.replace("pkl", "csv"), "wb") as OUTPUT:
        OUTPUT.write("user_id,chain\n")

        for user_id, d in booking_cluster_chain.items():
            od = collections.OrderedDict(sorted(d.items()))
            OUTPUT.write("{},{}\n".format(user_id, "-".join(od.values())))

            # For pickle file
            pure_cluster_chain[user_id] = [cluster[:cluster.find("(")] for cluster in od.values() if cluster.find("(") > -1]

    log("Write the booking chain file in {}".format(filepath_cluster_pkl), INFO)
    save_cache(pure_cluster_chain, filepath_cluster_pkl)

    return pure_cluster_chain

def srch_hotel_cluster(filepath_train, filepath_test, filepath_csv, datetime_idx=0, srch_distance_idx=6, srch_id_idx=16, hotel_country_idx=20, hotel_market_idx=21, hotel_idx=-1):
    global SPLIT_CHAR, UNKNOWN_HOTEL_CLUSTER

    srch_clusters = {}
    with open(filepath_train, "rb") as INPUT:
        for line in INPUT:
            arr = line.strip().split(SPLIT_CHAR)
            date, srch_id, srch_distance, hotel_cluster = arr[datetime_idx], arr[srch_id_idx], arr[srch_distance_idx], arr[hotel_idx]
            if not srch_id.isdigit():
                continue

            key = None
            if srch_distance != "":
                key = "{}-{}".format(srch_id, srch_distance)
            else:
                hotel_country, hotel_market = arr[hotel_country_idx], arr[hotel_market_idx]
                key = "{}-{};{}".format(srch_id, hotel_country, hotel_market)

            srch_clusters.setdefault(key, {})
            srch_clusters[key][date] = "{}".format(hotel_cluster)

    count_matching_distance, count_matching_hotel_arr, count_non_matching = 0, 0, 0
    with open(filepath_test, "rb") as INPUT:
        for line in INPUT:
            arr = line.strip().split(SPLIT_CHAR)
            date, srch_id, srch_distance = arr[datetime_idx+1], arr[srch_id_idx+1], arr[srch_distance_idx+1]
            if not srch_id.isdigit():
                continue

            key = None
            if srch_distance != "":
                key = "{}-{}".format(srch_id, srch_distance)

                if key in srch_clusters:
                    count_matching_distance += 1
                else:
                    count_non_matching += 1
            else:
                # Lack of is_booking and cnt fields
                hotel_country, hotel_market = arr[hotel_country_idx-1], arr[hotel_market_idx-1]
                key = "{}-{};{}".format(srch_id, hotel_country, hotel_market)

                if key in srch_clusters:
                    count_matching_hotel_arr += 1
                else:
                    count_non_matching += 1

            srch_clusters.setdefault(key, {})
            srch_clusters[key][date] = UNKNOWN_HOTEL_CLUSTER

    log("count_matching_distance: {}, count_matching_hotel_arr: {}, count_non_matching: {}".format(count_matching_distance, count_matching_hotel_arr, count_non_matching), INFO)

    pure_cluster = {}
    with open(filepath_csv + ".txt", "wb") as TXT:
        with open(filepath_csv + ".csv", "wb") as OUTPUT:
            OUTPUT.write("srch_destination_id,hotel_clusters\n")

            for key in sorted(srch_clusters.keys()):
                d = srch_clusters[key]

                od = collections.OrderedDict(sorted(d.items()))
                OUTPUT.write("{},{}\n".format(key, "-".join(od.values())))

                TXT.write("srch_id:{}\n".format(key))
                TXT.write("{}\n".format("-"*100))
                for date, hotel_cluster in od.items():
                    TXT.write("{}:{}\n".format(date, hotel_cluster))
                TXT.write("\n")

                pure_cluster[key] = od.values()

    save_cache(pure_cluster, filepath_csv+".pkl")

    return pure_cluster

if __name__ == "__main__":
    # Read Global Solutions
    #filepath_global = "/Users/RungChiChen/Documents/programs/kaggle/cases/Expedia Hotel Recommendations/input/train/train.csv"
    #global_solutions = global_solution(filepath_global)
    #log("Finish loading global solution from {}".format(filepath_global), INFO)

    # Booking Chain
    filepath_train = "/Users/RungChiChen/Documents/programs/kaggle/cases/Expedia Hotel Recommendations/input/train/train.csv"
    filepath_test = "/Users/RungChiChen/Documents/programs/kaggle/cases/Expedia Hotel Recommendations/input/test/test.csv"

    #filepath_chain = "/Users/RungChiChen/Documents/programs/kaggle/cases/Expedia Hotel Recommendations/input/chain=hotel_cluster.pkl"
    #booking_chain(filepath_train, filepath_test, filepath_chain, target_idx=-1)
    #log("Finish loading chain solution from {}".format(filepath_chain), INFO)

    #filepath_chain = "/Users/RungChiChen/Documents/programs/kaggle/cases/Expedia Hotel Recommendations/input/chain=srch_destinaion_id.pkl"
    #booking_chain(filepath_train, filepath_test, filepath_chain, target_idx=16)
    #log("Finish loading chain solution from {}".format(filepath_chain), INFO)

    filepath_srch_id = "/Users/RungChiChen/Documents/programs/kaggle/cases/Expedia Hotel Recommendations/input/base=srch_chain=cluster.pkl"
    booking_chain(filepath_train, filepath_test, filepath_srch_id, base_idx=16, target_idx=-1)

    sys.exit(1)

    filepath_csv = "/Users/RungChiChen/Documents/programs/kaggle/cases/Expedia Hotel Recommendations/input/srch_hotel_cluster"
    srch_hotel_cluster(filepath_train, filepath_test, filepath_csv)

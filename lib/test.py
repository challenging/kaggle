#!/usr/bin/python

import time

import glob

from joblib import Parallel, delayed

def read_single_file(filepath):
    results = []

    with open(filepath, "rb") as INPUT:
        print "Start to read {}".format(filepath)
        for line in INPUT:
            results.append(line.split(","))

    return results

def read_multi_file(f_pattern, n_jobs=4):
    results = []
    results.extend(Parallel(n_jobs=n_jobs)(delayed(read_single_file)(im_file) for im_file in glob.iglob(f_pattern)))

    return results

if __name__ == "__main__":
    timestamp_start = time.time()
    results = read_single_file("/Users/rongqichen/Documents/programs/kaggle/cases/Expedia Hotel Recommendations/input/train/train.csv")
    timestamp_end = time.time()
    print("Spend {:.4f} secends to read {} records".format(timestamp_end - timestamp_start, len(results)))

    timestamp_start = time.time()
    results = read_multi_file("/Users/rongqichen/Documents/programs/kaggle/cases/Expedia Hotel Recommendations/input/train/*part*", n_jobs=8)
    timestamp_end = time.time()
    print("Spend {:.4f} secends to read {} records".format(timestamp_end - timestamp_start, len(results)))

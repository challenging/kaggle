#!/usr/bin/env python

import sys

import json
import base64

import beanstalkc

IP_BEANSTALK, PORT_BEANSTALK = "127.0.0.1", 11300

if __name__ == "__main__":
    talk = beanstalkc.Connection(host=IP_BEANSTALK, port=PORT_BEANSTALK)
    task = "facebook_checkin_competition"

    string = {"method": "most_popular",
              "strategy": "median",
              "setting": {},
              "n_top": 10,
              "criteria": [4, 4],
              "is_normalization": False,
              "is_accuracy": False,
              "is_exclude_outlier": False,
              "is_testing": False,
              "cache_workspace": "/Users/rongqichen/Documents/programs/kaggle/cases/Facebook V - Predicting Check Ins/input/cache/criteria=512x1024_windowsize=0.1,0.1_batchsize=500000_isaccuracy=False_excludeoutlier=False_istesting=False/method=most_popular_strategy=native.8e959ea180d07fd2ec6f17ef00c9421b.10/99914b932bd37a50b983c5e7c90ae93b",
              "filepath_training": "/Users/rongqichen/Documents/programs/kaggle/cases/Facebook V - Predicting Check Ins/input/testing/train.csv",
              "filepath_testing": "/Users/rongqichen/Documents/programs/kaggle/cases/Facebook V - Predicting Check Ins/input/testing/test.csv"}

    e_string = json.dumps(string)
    request = base64.b64encode(e_string)
    talk.use(task)

    talk.put(request)
    print "Push {} into the {} queue".format(e_string, task)

    talk.close()

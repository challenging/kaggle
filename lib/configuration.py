#!/usr/bin/env python

import re
import copy
import ConfigParser

import numpy as np

class FacebookConfiguration(object):
    def __init__(self, filepath):
        self.config = ConfigParser.RawConfigParser()
        self.config.read(filepath)

    def get_workspace(self):
        return self.config.get("MAIN", "workspace")

    def is_accuracy(self):
        is_accuracy = self.config.getint("FEATURE", "is_accuracy")

        return True if is_accuracy == 1 else False

    def is_exclude_outlier(self):
        return True if self.config.getint("FEATURE", "is_exclude_outlier") == 1 else False

class ModelConfParser(object):
    def __init__(self, filepath):
        self.config = ConfigParser.RawConfigParser()
        self.config.read(filepath)

    def get_workspace(self):
        return self.config.get("MAIN", "workspace")

    def get_objective(self):
        return self.config.get("MAIN", "objective")

    def get_cost(self):
        return self.config.get("MAIN", "cost")

    def get_nfold(self):
        nfold = 10

        if self.config.has_option("MAIN", "nfold"):
            nfold = self.config.getint("MAIN", "nfold")

        return nfold

    def get_n_jobs(self):
        if self.config.has_option("MAIN", "n_jobs"):
            return self.config.getint("MAIN", "n_jobs")
        else:
            return -1

    def get_top_feature(self):
        top_feature = 512

        if self.config.has_option("MAIN", "top_feature"):
            top_feature = self.config.getint("MAIN", "top_feature")

        return top_feature

    def get_n_estimators(self):
        value = 1000

        if self.config.has_option("MAIN", "n_estimators"):
            value = self.config.getint("MAIN", "n_estimators")

        return value

    def get_learning_rate(self):
        value = 0.05

        if self.config.has_option("MAIN", "learning_rate"):
            value = self.config.getfloat("MAIN", "learning_rate")

        return value

    def get_interaction_information(self):
        return self.config.getint("INTERACTION_INFORMATION", "binsize"), self.config.get("INTERACTION_INFORMATION", "top")

    def get_global_setting(self):
        workspace, nfold, cost_string = self.config.get("MAIN", "workspace"), self.config.get("MAIN", "nfold"), self.config.get("MAIN", "cost")

        return workspace, nfold, cust_string

    def get_model_setting(self, model_section):
        d = {}
        for option in self.config.options(model_section):
            v = self.config.get(model_section, option).strip("\"")

            if option == "class_weight":
                if v.isdigit():
                    d["class_weight"] = {0: int(v), 1: 1}
                else:
                    try:
                        d["class_weight"] = {0: float(v), 1: 1}
                    except Exception as e:
                        setting["class_weight"] = "balanced"
            elif v.isdigit():
                d[option.lower()] = int(v)
            elif v == "nan":
                d[option.lower()] = np.nan
            elif v.lower() in ["true", "false"]:
                d[option.lower()] = (v.lower() == "true")
            else:
                try:
                    d[option.lower()] = float(v)
                except:
                    d[option.lower()] = v

        method = d.pop("method")

        if method.find("deep") == -1:
            d.setdefault("n_jobs", self.get_n_jobs())

        if "kernal" in d:
            d["method"] = d.pop("kernal")

        if "calibration" in d:
            calibration = d.pop("calibration")

            if calibration == 1:
                calibration_method = re.sub("(randomforest|extratree|xgboosting)", "calibration", method)
                calibration_setting = {}
                calibration_setting["n_jobs"] = -1
                calibration_setting["cv"] = self.get_nfold()
                calibration_setting["dependency"] = (method, copy.deepcopy(d))

                for key in ["model_id"]:
                    if key in calibration_setting["dependency"][1]:
                        del calibration_setting["dependency"][1][key]

                calibration_setting["data_dimension"] = calibration_setting["dependency"][1].pop("data_dimension")
                calibration_setting["method"] = "isotonic"

                yield (calibration_method, calibration_setting)

        yield (method, d)

    def get_layer_models(self, layer_number):
        for section in self.config.sections():
            if section.find("LAYER{}_".format(layer_number)) > -1:
                yield section

if __name__ == "__main__":
    parser = ModelConfParser("../etc/conf/Santander.cfg")
    for model_section in parser.get_layer_models(2):
        for method, setting in parser.get_model_setting(model_section):
            print method, setting

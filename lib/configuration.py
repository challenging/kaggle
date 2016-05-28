#!/usr/bin/env python

import re
import copy
import getpass
import ConfigParser

import numpy as np

from utils import make_a_stamp

MAIN = "MAIN"

FEATURE_IMPORTANCE = "FEATURE_IMPORTANCE"
FEATURE_INTERACTION = "FEATURE_INTERACTION"

OPTION_IS_FULL = "is_full"

class KaggleConfiguration(object):
    def __init__(self, filepath):
        self.config = ConfigParser.RawConfigParser()
        self.config.read(filepath)

    def get_value(self, section, option):
        value = None
        if self.config.has_option(section, option):
            value = self.config.get(section, option)

        return value

class FacebookConfiguration(KaggleConfiguration):
    def get_workspace(self, section):
        workspace, cache_workspace, submission_workspace = self.config.get(section, "workspace"), self.config.get(MAIN, "cache_workspace"), self.config.get(MAIN, "output_workspace")

        if getpass.getuser() != "RungChiChen":
            for folder in [workspace, cache_workspace, submission_workspace]:
                folder = folder.replace("RungChiChen", "rongqichen")

        return workspace, cache_workspace, submission_workspace

    def get_methods(self):
        sections = []
        for section in self.config.sections():
            if section.find("METHOD") > -1 and section.find("-") == -1:
                sections.append(section)

        return sorted(sections)

    def get_setting(self, section):
        setting = {}

        if self.config.has_section(section):
            setting = dict(self.config.items(section))
            for key, value in setting.items():
                if value.isdigit():
                    setting[key] = int(value)
                else:
                    setting[key] = float(value)

        return setting

    def get_method_detail(self, section):
        method, criteria, strategy = "most_popular", ("4096", "4096"), "native"

        if self.config.has_option(section, "name"):
            method = self.config.get(section, "name")

        if self.config.has_option(section, "criteria"):
            if method == "most_popular":
                criteria = self.config.get(section, "criteria").split(",")

        if method == "kdtree":
            strategy = self.config.get(section, "strategy")

        return method, criteria, strategy, self.get_stamp(section), self.get_size(section), self.is_accuracy(section), self.is_exclude_outlier(section)

    def get_stamp(self, section):
        names = self.get_workspace(section)[0].split("/")

        turn_on, pool = 0, []
        for name in names:
            if name == "input":
                turn_on = 1

            if turn_on == 1:
                pool.append(name)

        return make_a_stamp(pool)

    def get_size(self, section):
        window_size = "all"
        t = re.search("windown_size=([\d\.,]+)", self.get_workspace(section)[0])
        if t:
            window_size = t.groups(0)[0]

        batch_size = 10000
        if self.config.has_option(section, "batch_size"):
            batch_size = self.config.getint(section, "batch_size")

        n_top = 3
        if self.config.has_option(section, "n_top"):
            n_top = self.config.getint(section, "n_top")

        return window_size, batch_size, n_top

    def is_accuracy(self, section, option="is_accuracy"):
        return self.is_exclude_outlier(section, option)

    def is_exclude_outlier(self, section, option="is_exclude_outlier"):
        return True if self.get_value(section, option) and self.get_value(section, option) == "1" else False

    def is_full(self, section=MAIN, option=OPTION_IS_FULL):
        return self.get_value(section, option)

    def get_weight(self, section, option="weight"):
        return float(self.get_value(section, option))

class ModelConfParser(KaggleConfiguration):
    def get_filepaths(self, method):
        return self.get_value(MAIN, "filepath_training"), self.get_value(MAIN, "filepath_testing"), self.get_value(MAIN, "filepath_submission").format(method=method), self.get_value(MAIN, "filepath_tuning").format(method=method)

    def get_workspace(self):
        return self.config.get(MAIN, "workspace")

    def get_objective(self):
        return self.config.get(MAIN, "objective")

    def get_cost(self):
        return self.config.get(MAIN, "cost")

    def get_nfold(self):
        nfold = 10

        if self.config.has_option(MAIN, "nfold"):
            nfold = self.config.getint(MAIN, "nfold")

        return nfold

    def get_n_jobs(self):
        if self.config.has_option(MAIN, "n_jobs"):
            return self.config.getint(MAIN, "n_jobs")
        else:
            return -1

    def get_top_feature(self):
        top_feature = None

        if self.config.has_option(MAIN, "top_feature"):
            top_feature = self.config.getint(MAIN, "top_feature")

        return top_feature

    def get_n_estimators(self):
        value = 1000

        if self.config.has_option(MAIN, "n_estimators"):
            value = self.config.getint(MAIN, "n_estimators")

        return value

    def get_learning_rate(self):
        value = 0.05

        if self.config.has_option(MAIN, "learning_rate"):
            value = self.config.getfloat(MAIN, "learning_rate")

        return value

    def get_feature_interaction(self):
        return self.get_value(FEATURE_INTERACTION, "filepath"), self.get_value(FEATURE_INTERACTION, "binsize"), self.get_value(FEATURE_INTERACTION, "top")

    def get_feature_importance(self):
        return self.get_value(FEATURE_IMPORTANCE, "filepath"), self.get_value(FEATURE_IMPORTANCE, "top")

    def get_global_setting(self):
        workspace, nfold, cost_string = self.config.get(MAIN, "workspace"), self.config.get(MAIN, "nfold"), self.config.get(MAIN, "cost")

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

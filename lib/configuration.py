#!/usr/bin/env python

import copy
import ConfigParser

import numpy as np

class ModelConfParser(object):
    def __init__(self, filepath):
        self.config = ConfigParser.RawConfigParser()
        self.config.read(filepath)

        self.models = {}

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

        if "model_id" in d:
            self.models[d.pop("model_id")] = (method, copy.deepcopy(d))

        if "dependency_model_id" in d:
            d["dependency"] = self.models[d.pop("dependency_model_id")]

            for key in ["data_dimension", "model_id"]:
                if key in d["dependency"][1]:
                    del d["dependency"][1][key]

        return method, d

    def get_layer_models(self, layer_number):
        for section in self.config.sections():
            if section.find("LAYER{}_".format(layer_number)) > -1:
                yield section

if __name__ == "__main__":
    parser = ModelConfParser("../etc/conf/Santander_testing.cfg")
    for model_section in parser.get_layer_models(1):
        cfg = parser.get_model_setting(model_section)
        print cfg

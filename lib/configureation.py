#!/usr/bin/env python

import ConfigParser

class ModelConfParser(object):
    def __init__(self, filepath):
        self.config = ConfigParser.RawConfigParser()
        self.config.read(filepath)

    def get_global_setting(self):
        workspace, nfold, cost_string = self.config.get("MAIN", "workspace"), self.config.get("MAIN", "nfold"), self.config.get("MAIN", "cost")

        return workspace, nfold, cust_string

    def get_model_setting(self, model_section):
        d = {}
        for option in self.config.options(model_section):
            v = self.config.get(model_section, option)
            if v.isdigit():
                d[option.lower()] = int(v)
            else:
                try:
                    d[option.lower()] = float(v)
                except:
                    d[option.lower()] = v

        return d

    def get_layer_models(self):
        for section in self.config.sections():
            if section.find("LAYER1_") > -1:
                yield section

if __name__ == "__main__":
    parser = ModelConfParser("../etc/conf/BNP.cfg")
    for model_section in parser.get_layer_models():
        cfg = parser.get_model_setting(model_section)
        print cfg

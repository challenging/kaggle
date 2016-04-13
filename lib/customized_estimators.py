#!/usr/bin/env python

import os
import sys

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.metrics import log_loss

from scipy.optimize import minimize

from utils import log, INFO, WARN

class CustomizedClassEstimator(BaseEstimator):
    def __init__(self, cost_func, n_class=2):
        BaseEstimator.__init__(self)
        self.n_class = n_class
        self.cost_func = cost_func

    @staticmethod
    def function(w, xs, y, cost_func=log_loss, n_class=2):
        w = np.abs(w)
        sol = np.zeros(xs[0].shape)

        for idx, wi in enumerate(w):
            sol += xs[idx] * wi

        score = cost_func(y, sol)

        return score

    def fit(self, x, y):
        xs = np.hsplit(x, x.shape[1]/self.n_class)
        x0 = np.ones(len(xs)) / float(len(xs))

        bounds = [(0, 1)]*len(x0)   # weight should be bounded in (0, 1)
        cons = ({"type": "eq", "fun": lambda w: 1-sum(w)})  # the sum of weight should be 1

        res = minimize(CustomizedClassEstimator.function,
                       x0,
                       args=(xs, y, self.cost_func, self.n_class),
                       bounds=bounds,
                       constraints=cons)

        self.w = res.x

        return self

    def predict_proba(self, x):
        xs = np.hsplit(x, x.shape[1]/self.n_class)

        y_pred = np.zeros(xs[0].shape)
        for idx, wi in enumerate(self.w):
            y_pred += xs[idx]*wi

        return y_pred

class CustomizedProbaEstimator(BaseEstimator):
    def __init__(self, cost_func, n_class=2):
        BaseEstimator.__init__(self)
        self.n_class = n_class
        self.cost_func = cost_func

    @staticmethod
    def function(w, xs, y, cost_func=log_loss, n_class=2):
        w = w / w.sum()

        sol = np.zeros(xs[0].shape)
        for i, wi in enumerate(w):
            sol[:, i%n_class] += xs[int(i/n_class)][:, i%n_class]*wi

        score = cost_func(y, sol)
        return score

    def fit(self, x, y):
        xs = np.hsplit(x, x.shape[1]/self.n_class)
        x0 = np.ones(len(xs)*self.n_class) / float(len(xs))

        bounds = [(0, 1)]*len(x0)

        res = minimize(CustomizedProbaEstimator.function,
                       x0,
                       args=(xs, y, self.cost_func, self.n_class),
                       bounds=bounds)

        self.w = res.x

        return self

    def predict_proba(self, x):
        xs = np.hsplit(x, x.shape[1]/self.n_class)

        y_pred = np.zeros(xs[0].shape)
        for idx, wi in enumerate(self.w):
            y_pred[:, idx%self.n_class] += xs[int(idx/self.n_class)][:i%self.n_class]*wi

        return y_pred

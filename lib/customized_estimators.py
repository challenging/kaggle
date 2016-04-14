#!/usr/bin/env python

import os
import sys

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import log_loss

from scipy.optimize import minimize

from utils import log, INFO, WARN

class FinalEnsembleModel(BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier for scikit-learn estimators.

    Parameters
    ----------

    clf : `iterable`
      A list of scikit-learn classifier objects.
    weights : `list` (default: `None`)
      If `None`, the majority rule voting will be applied to the predicted class labels.
        If a list of weights (`float` or `int`) is provided, the averaged raw probabilities (via `predict_proba`)
        will be used to determine the most confident class label.

    """
    def __init__(self, clfs, weights=None):
        self.clfs = clfs
        self.weights = weights

    def fit(self, X, y):
        """
        Fit the scikit-learn estimators.

        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]
            Training data
        y : list or numpy array, shape = [n_samples]
            Class labels

        """
        for clf in self.clfs:
            clf.fit(X, y)

    def predict(self, X):
        """
        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]

        Returns
        ----------

        maj : list or numpy array, shape = [n_samples]
            Predicted class labels by majority rule

        """

        self.classes_ = np.asarray([clf.predict(X) for clf in self.clfs])
        if self.weights:
            avg = self.predict_proba(X)

            maj = np.apply_along_axis(lambda x: max(enumerate(x), key=operator.itemgetter(1))[0], axis=1, arr=avg)

        else:
            maj = np.asarray([np.argmax(np.bincount(self.classes_[:,c])) for c in range(self.classes_.shape[1])])

        return maj

    def predict_proba(self, X):
        """
        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]

        Returns
        ----------

        avg : list or numpy array, shape = [n_samples, n_probabilities]
            Weighted average probability for each class per sample.

        """
        self.probas_ = [clf.predict_proba(X) for clf in self.clfs]
        avg = np.average(self.probas_, axis=0, weights=self.weights)

        return avg

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

    def get_weights(self):
        return self.w

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

    def get_weights(self):
        return self.w

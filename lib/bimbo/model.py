#!/usr/bin/env python

import threading
import Queue

import numpy as np

from sklearn.cross_validation import KFold

from utils import log
from utils import DEBUG, INFO, WARN, ERROR

class LearningLayer(object):
    def __init__(self, dataset_x, dataset_y, predicted_x, models, nfold=3):
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        self.predicted_x = predicted_x

        self.models = models
        self.nfold = nfold

        self.results_testing = np.zeros((self.dataset_x.shape[0], len(self.models)))
        self.results = np.zeros((self.predicted_x.shape[0], len(self.models)))

        self.queue = Queue.Queue()

        self.__nfold()

    def __nfold(self):
        skf = KFold(self.dataset_y.shape[0], self.nfold, shuffle=True, random_state=1201)

        for nfold, (train, test) in enumerate(skf):
            for model_idx, model in enumerate(self.models):
                self.queue.put(((train, test), nfold, model_idx, model))

    def get_results(self):
        return self.results_testing.mean(axis=1)

    def predict(self):
        return self.results.mean(axis=1) / self.nfold

    def raw_results(self):
        return self.results_testing

class LearningChunk(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        threading.Thread.__init__(self, group=group, target=target, name=name, verbose=verbose)

        self.args = args

        for key, value in kwargs.items():
            setattr(self, key, value)

    def run(self):
        while True:
            (train_idx, test_idx), nfold, model_idx, model = self.queue.get()
            log("Start to train {}-th model for {}-th nfold".format(model_idx+1, nfold+1), DEBUG)

            model.fit(self.dataset_x[train_idx], self.dataset_y[train_idx])

            self.results_testing[test_idx,model_idx] = model.predict(self.dataset_x[test_idx])
            self.results[:,model_idx] += model.predict(self.predicted_x)

            self.queue.task_done()

class Learning(object):
    def __init__(self, dataset_x, dataset_y, models, nfold=3, n_jobs=1):
        self.layer = LearningLayer(dataset_x, dataset_y, models, nfold)

        for n in range(0, n_jobs):
            thread = LearningChunk(kwargs={"queue": self.layer.queue,
                                           "results_testing": self.layer.results_testing, "results": self.layer.results,
                                           "dataset_x": self.layer.dataset_x, "dataset_y": self.layer.dataset_y,
                                           "predicted_x": self.layer.predicted_x})
            thread.setDaemon(True)
            thread.start()

        self.layer.queue.join()

    def get_models(self):
        return self.layer.models

    def get_results(self, is_model=False):
        if is_model:
            return [self.layer.raw_results()[:,model_idx] for model_idx in range(0, len(self.layer.models))]
        else:
            return self.layer.get_results()

    def predict(self):
        return self.layer.predict()

class LearningCost(object):
    @staticmethod
    def rmsle_1(true_set, predicted_set):
        error_square = 0
        for t, p in zip(true_set, predicted_set):
            error_square += (np.log1p(t)-np.log1p(p))**2

        return np.sqrt(error_square/len(true_set))

    @staticmethod
    def rmsle_2(true_set, predicted_set):
        error_square = 0
        for t, p in zip(true_set, predicted_set):
            error_square += (t-p)**2

        return np.sqrt(error_square/len(true_set))

if __name__ == "__main__":
    from sklearn import linear_model

    dataset_x = np.array([[0, 0], [1, 0], [0, 1], [4, 2], [8, 2], [1, 5]])
    dataset_y = np.array([0, 0.5, 0.5, 3, 5, 3])

    predicted_x = np.array([[0, 10], [11, 0], [101, 1], [24, 2]])

    models = [linear_model.LinearRegression(), linear_model.Ridge(alpha=0.5)]
    nfold = 3

    learning = Learning(dataset_x, dataset_y, predicted_x, models, nfold)
    for clf in learning.get_models():
        log("The coef is {}".format(clf.coef_), INFO)

    model_results = learning.get_results(True)
    for model_idx in range(0, len(models)):
        log("{} - {}".format(model_idx+1, model_results[model_idx]), INFO)

    log("The cost is {}".format(LearningCost.rmsle_1(dataset_y, learning.get_results(False))), INFO)
    log("The predicted results is {}".format(learning.predict()), INFO)

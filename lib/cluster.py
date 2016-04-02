#!/usr/bin/env python

import numpy as np

from utils import log, INFO, WARN
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, AgglomerativeClustering, SpectralClustering
from sklearn.neighbors import NearestCentroid, kneighbors_graph
from sklearn.preprocessing import MinMaxScaler, Imputer, StandardScaler

class ClusterFactory(object):
    @staticmethod
    def get_model(method, setting):
        if method.find("kmeans") > -1:
            return Cluster(method, KMeans(n_clusters=setting["n_clusters"], n_init=setting.get("n_init", 10), init="k-means++", random_state=1201, n_jobs=setting.get("n_jobs", 1)))
        elif method.find("knn") > -1:
            return Cluster(method, NearestCentroid(shrink_threshold=setting.get("shrink", 0.1)))
        elif method.find("affinitypropagation") > -1:
            return Cluster(method, AffinityPropagation(convergence_iter=setting.get("convergence_iter", 100), max_iter=setting.get("max_iter", 5000), verbose=1))
        elif method.find("spectral") > -1:
            return Cluster(method, SpectralClustering(n_clusters=setting["n_clusters"], eigen_solver='amg', affinity="rbf"))
        elif method.find("agglomerative") > -1:
            # connectivity matrix for structured Ward
            connectivity = kneighbors_graph(setting["X"], n_neighbors=setting.get("n_neighbors", 10), include_self=False)
            # make connectivity symmetric
            connectivity = 0.5 * (connectivity + connectivity.T)

            return Cluster(method, AgglomerativeClustering(linkage="average", affinity="cityblock", n_clusters=setting["n_clusters"], connectivity=connectivity))

class Cluster(object):
    def __init__(self, name, model):
        self.name = name
        self.model = model

    def fit(self, train_x, train_y=None, is_norm=True):
        # Normalization
        if is_norm:
            train_x_min = train_x.min(0)
            train_x_ptp = train_x.ptp(axis=0)

            train_x = train_x.astype(float) - train_x_min / train_x_ptp

            if np.any(train_y):
                train_y = train_y.astype(float) - train_x_min / train_x_ptp

        imp = Imputer(missing_values='NaN', strategy='mean', axis=1)
        imp.fit(train_x)
        if np.isnan(train_x).any():
            log("Found {} NaN values in train_x, so try to transform them to 'mean'".format(np.isnan(train_x).sum()), WARN)
            train_x = imp.transform(train_x)

        if np.any(train_y) and np.isnan(train_y).any():
            log("Found {} NaN values in train_y, so try to transform them to 'mean'".format(np.isnan(train_y).sum()), WARN)
            train_y = imp.transform(train_y)

        if np.any(train_y):
            self.model.fit(train_x, train_y)
        else:
            self.model.fit(train_x)

    def predict(self, data):
        return self.model.predict(data)

    def get_centroid(self):
        return self.model.cluster_centers_

    def get_labels(self):
        return self.model.labels_

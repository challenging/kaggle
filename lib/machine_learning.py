#!/usr/bin/env python

import time

import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble
from sklearn.metrics import log_loss, make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split

import xgboost as xgb

import random
random.seed(1201)

def extra_tree_classifier(train_x, train_y):
    print('Training...')
    clf = ExtraTreesClassifier(n_estimators=850,max_features= 60,criterion= 'entropy',min_samples_split= 4,
                               max_depth= 40, min_samples_leaf= 2, n_jobs = -1)
    clf.fit(train_x, train_y)

    return clf

def flog_loss(ground_truth, predictions):
    flog_loss_ = log_loss(ground_truth, predictions) #, eps=1e-15, normalize=True, sample_weight=None)
    return flog_loss_

def ensemble_probability(train_x, train_y, test, ne=25, lr=1e-2):
    LL  = make_scorer(flog_loss, greater_is_better=False)

    g = {'ne':ne, 'md':40, 'mf':60, 'rs': 1201}
    etc = ensemble.ExtraTreesClassifier(n_estimators=g['ne'], max_depth=g['md'], max_features=g['mf'], random_state=g['rs'],
                                        criterion='entropy', min_samples_split=4, min_samples_leaf=2, verbose=0, n_jobs=-1)
    etr = ensemble.ExtraTreesRegressor(n_estimators=g['ne'], max_depth=g['md'], max_features=g['mf'], random_state=g['rs'],
                                       min_samples_split=4, min_samples_leaf=2, verbose=0, n_jobs =-1)

    rfc = ensemble.RandomForestClassifier(n_estimators=g['ne'], max_depth=g['md'], max_features=g['mf'], random_state=g['rs'],
                                          criterion='entropy', min_samples_split=4, min_samples_leaf=2, verbose=0, n_jobs=-1)
    rfr = ensemble.RandomForestRegressor(n_estimators=g['ne'], max_depth=g['md'], max_features=g['mf'], random_state=g['rs'],
                                         min_samples_split= 4, min_samples_leaf= 2, verbose = 0, n_jobs =-1)

    xgr = xgb.XGBRegressor(n_estimators=g['ne'], max_depth=g['md'], seed=g['rs'], missing=np.nan, learning_rate=lr, subsample=0.8, colsample_bytree=0.85)
    xgc = xgb.XGBClassifier(n_estimators=g['ne'], max_depth=g['md'], seed=g['rs'], missing=np.nan, learning_rate=lr, subsample=0.8, colsample_bytree=0.85)

    clf = {'etc':etc, 'etr':etr, 'rfc':rfc, 'rfr':rfr, 'xgr':xgr, 'xgc':xgc}
    y_pred = {}
    for key in clf.keys():
        y_pred[key] = []

    best_score, start_time = 0.0, time.time()

    print "Training..."
    for c in clf.keys():
        if c[:1] != "x": #not xgb
            model = GridSearchCV(estimator=clf[c], param_grid={}, n_jobs =-1, cv=2, verbose=0, scoring=LL)
            model.fit(train_x, train_y)
            if c[-1:] != "c": #not classifier
                y_pred[c] = model.predict(test)
                print("Ensemble Model: ", c, " Best CV score: ", model.best_score_, " Time: ", round(((time.time() - start_time)/60),2))
            else: #classifier
                best_score = (log_loss(train_y, model.predict_proba(train_x)[:,1]))*-1
                y_pred[c] = model.predict_proba(test)[:,1]
                print("Ensemble Model: ", c, " Best CV score: ", best_score, " Time: ", round(((time.time() - start_time)/60),2))
        else: #xgb
            X_fit, X_eval, y_fit, y_eval= train_test_split(train_x, train_y, test_size=0.35, train_size=0.65, random_state=g['rs'])
            model = clf[c]
            model.fit(X_fit, y_fit, early_stopping_rounds=20, eval_metric="logloss", eval_set=[(X_eval, y_eval)], verbose=0)
            if c == "xgr": #xgb regressor
                best_score = (log_loss(train_y, model.predict(train_x)))*-1
                y_pred[c] = model.predict(test)
            else: #xgb classifier
                best_score = (log_loss(train_y, model.predict_proba(train_x)[:,1]))*-1
                y_pred[c] = model.predict_proba(test)[:,1]
            print("Ensemble Model: ", c, " Best CV score: ", best_score, " Time: ", round(((time.time() - start_time)/60),2))

        for i in range(len(y_pred[c])):
            if y_pred[c][i] < 0.0:
                y_pred[c][i] = 0.0
            if y_pred[c][i] > 1.0:
                y_pred[c][i] = 1.0

    return y_pred

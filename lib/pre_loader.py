#!/usr/bin/env python

import sys
import datetime

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Merge
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint

from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction import DictVectorizer

def expedia_loader(filepath_train, filepath_test, filepath_destination,
    drop_fields=["date_time", "site_name","posa_continent","user_id","is_mobile","is_package","channel","srch_ci","srch_co","hotel_continent", "hotel_country", 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', 'srch_destination_type_id']):

    def transform(df):
        columns = df.columns.values
        for column in columns:
            if column.lower() == "orig_destination_distance":
                df[column] = df[column].astype(np.float64)
            elif column.lower().find("d") == 0:
                df[column] = df[column].astype(np.float64)
            elif column.lower().find("cnt") == -1:
                df[column] = df[column].astype(str)

        return df

    df_destination = pd.read_csv(filepath_destination, index_col="srch_destination_id")

    df_train = pd.read_csv(filepath_train, index_col="srch_destination_id")
    #df_train["date_time"] = df_train["date_time"].map(lambda x: str(datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").weekday()))
    hotel_cluster = df_train["hotel_cluster"].values
    df_train = pd.merge(df_train, df_destination, left_index=True, right_index=True, how="left", sort=False)
    df_train = df_train.reset_index()
    df_train = df_train.drop(drop_fields + ["cnt"] + ["hotel_cluster"] + ["is_booking"], axis=1)

    df_test = pd.read_csv(filepath_test, index_col="srch_destination_id")
    #df_test["date_time"] = df_test["date_time"].map(lambda x: str(datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").weekday()))
    test_id = df_test["id"].values
    df_test = pd.merge(df_test, df_destination, left_index=True, right_index=True, how="left", sort=False)
    df_test = df_test.reset_index()
    df_test = df_test.drop(drop_fields + ["id"], axis=1)

    df_train = df_train.fillna(df_train.mean().fillna(0))
    df_test = df_test.fillna(df_test.mean().fillna(0))

    return transform(df_train), hotel_cluster, transform(df_test), test_id

def onehot_encoder(df):
    d = df.T.to_dict().values()

    dv = DictVectorizer()
    dv.fit(d)

    return dv

def get_onehot_transformer(train_x, test_x):
    df = pd.concat([train_x, test_x])

    return onehot_encoder(df)

def get_pca_transformer(train_x, train_y, n_components=-1):
    if n_components == -1:
        n_components = int(np.ceil(np.sqrt(train_x.shape[1])))

    pca = PCA(n_components=n_components)
    selection = SelectKBest(k=n_components/2)

    combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

    return combined_features.fit(train_x, train_y)

def data_process(filepath_train, filepath_test, filepath_destination):
    df_train, hotel_cluster, df_test, test_id = expedia_loader(filepath_train, filepath_test, filepath_destination)

    print df_train.columns[:30]

    onehot_transformer = get_onehot_transformer(df_train, df_test)
    training_dataset = onehot_transformer.transform(df_train.T.to_dict().values()).toarray()
    testing_dataset = onehot_transformer.transform(df_test.T.to_dict().values()).toarray()

    train_y = np.zeros((len(hotel_cluster), np.max(hotel_cluster)+1))
    for pos in hotel_cluster:
        train_y[pos] = 1

    pca_transformer = get_pca_transformer(training_dataset, hotel_cluster)
    pca_training_dataset = pca_transformer.transform(training_dataset)
    pca_testing_dataset = pca_transformer.transform(testing_dataset)

    return pca_training_dataset, train_y, pca_testing_dataset, test_id

def get_mlp_model(user_dims, dimension, output_dims, cost="categorical_crossentropy", learning_rate=1e-12, dropout_rate=0.5, init="uniform", activation="tanh"):
    model = Sequential()
    model.add(Dense(dimension, input_dim=user_dims[1], init=init, activation=activation))
    model.add(Dropout(dropout_rate))

    model.add(Dense(dimension, input_dim=dimension, init=init, activation=activation))
    model.add(Dropout(dropout_rate))

    model.add(Dense(dimension, input_dim=dimension, init=init, activation=activation))
    model.add(Dropout(dropout_rate))

    model.add(Dense(output_dims, activation='softmax'))

    sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=cost, optimizer=sgd, metrics=['accuracy'])

    return model

def mlp_model(user_dataset, train_y, test_x, dimension, nepoch=1024, batch_size=32):
    model= get_mlp_model(user_dataset.shape, dimension, train_y.shape[1])

    # Model training
    model.fit(user_dataset, train_y, nb_epoch=nepoch, batch_size=batch_size)

    # Prediction
    proba = model.predict_proba(test_x)

    return proba

def main(filepath_train, filepath_test, filepath_destination, dimension=32):
    train_x, train_y, test_x, test_id = data_process(filepath_train, filepath_test, filepath_destination)
    print train_x.shape, train_y.shape

    proba = mlp_model(train_x, train_y, test_x, dimension)
    print proba

if __name__ == "__main__":
    filepath_destination = "/Users/RungChiChen/Documents/programs/kaggle/cases/Expedia Hotel Recommendations/input/destinations.csv"
    filepath_train = "/Users/RungChiChen/Documents/programs/kaggle/cases/Expedia Hotel Recommendations/input/train/user_location_country/train_user_location_country=99.csv"
    filepath_test = "/Users/RungChiChen/Documents/programs/kaggle/cases/Expedia Hotel Recommendations/input/test/user_location_country/test_user_location_country=99.csv"

    main(filepath_train, filepath_test, filepath_destination)

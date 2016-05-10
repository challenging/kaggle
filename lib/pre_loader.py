#!/usr/bin/env python

import sys
import datetime

import pandas as pd

from sklearn.feature_extraction import DictVectorizer

def expedia_loader(filepath_train, filepath_test, drop_fields=["site_name","posa_continent","user_location_region","user_location_city","user_id","is_mobile","is_package","channel","srch_ci","srch_co","srch_destination_type_id","hotel_continent"]):
    def transform(df):
        columns = df.columns.values
        for column in columns:
            if column.lower() == "orig_destination_distance":
                df[column] = df[column].astype(float)
                df[column].fillna(-1)
            elif column.lower().find("cnt") == -1:
                df[column] = df[column].astype(str)

        return df

    df_train = pd.read_csv(filepath_train)
    df_train["date_time"] = df_train["date_time"].map(lambda x: str(datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").weekday()))
    hotel_cluster = df_train["hotel_cluster"].values
    df_train = df_train.drop(drop_fields + ["cnt"] + ["hotel_cluster"], axis=1)

    df_test = pd.read_csv(filepath_test)
    df_test["date_time"] = df_test["date_time"].map(lambda x: str(datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").weekday()))
    test_id = df_test["id"].values
    df_test = df_test.drop(drop_fields + ["id"], axis=1)

    return transform(df_train), hotel_cluster, transform(df_test), test_id

def onehot_encoder(df):
    d = df.T.to_dict().values()

    dv = DictVectorizer()
    transform_x = dv.fit_transform(d)

    return transform_x

if __name__ == "__main__":
    filepath_train = "/Users/RungChiChen/Documents/programs/kaggle/cases/Expedia Hotel Recommendations/input/train/user_location_country/train_user_location_country=205.csv"
    filepath_test = "/Users/RungChiChen/Documents/programs/kaggle/cases/Expedia Hotel Recommendations/input/test/user_location_country/test_user_location_country=205.csv"

    df_training, hotel_cluster, df_testing, test_id = expedia_loader(filepath_train, filepath_test)

    testing_x = onehot_encoder(df_testing)
    print testing_x.shape

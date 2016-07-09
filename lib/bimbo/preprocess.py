#!/usr/bin/env python

import os
import sys
import click

import pandas as pd

from utils import create_folder, log, INFO

WORKSPACE = "/Users/rongqichen/Documents/programs/kaggle/cases/Grupo Bimbo Inventory Demand/input"
TRAIN_FILE = os.path.join(WORKSPACE, "train.csv")
TEST_FILE = os.path.join(WORKSPACE, "test.csv")

# Semana,Agencia_ID,Canal_ID,Ruta_SAK,Cliente_ID,Producto_ID,Venta_uni_hoy,Venta_hoy,Dev_uni_proxima,Dev_proxima,Demanda_uni_equil

@click.command()
@click.option("--idx", required=True, help="column name for split")
def preprocess(idx):
    global TRAIN_FILE, TEST_FILE

    for filepath in [TRAIN_FILE, TEST_FILE]:
        df = pd.read_csv(filepath)

        for column in idx.split(","):
            output_folder = os.path.join(WORKSPACE, "split", column, os.path.basename(filepath).replace(".csv", ""))
            create_folder(os.path.join(output_folder, "1.txt"))

            for unique_value in df[column].unique():
                output_filepath = os.path.join(output_folder, "{}.csv".format(unique_value))

                row_ids = df[column] == unique_value

                df[row_ids].to_csv(output_filepath, index=False)

if __name__ == "__main__":
    preprocess()

import os
import pandas as pd
import sqlite3
import requests
from zipfile import ZipFile
from io import BytesIO
from kaggle.api.kaggle_api_extended import KaggleApi

# Kaggle dataset urls
dataset_1 = "nsidcorg/daily-sea-ice-extent-data"
dataset_2 = "sevgisarac/temperature-change"

data_path = "../data"
os.makedirs(data_path, exist_ok=True)


def download_kaggle_dataset(dataset, data_path):
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=data_path, unzip=True)

def pipeline(datasets, db_name, table_names):
    # Download datasets from Kaggle
    for dataset in datasets:
        download_kaggle_dataset(dataset, data_path)

    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    if len(csv_files) < 2:
        raise ValueError("Not enough CSV files.")

    df1 = pd.read_csv(os.path.join(data_path, csv_files[0]), encoding='cp1252')
    df2 = pd.read_csv(os.path.join(data_path, csv_files[3]))

    db_path = os.path.join(data_path, db_name)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    df1.to_sql(table_names[0], conn, if_exists="replace", index=False)
    df2.to_sql(table_names[1], conn, if_exists="replace", index=False)
    return df1, df2

if __name__ == "__main__":
    pipeline([dataset_1, dataset_2], "iceTemp2.db", ["ice", "temperature"])

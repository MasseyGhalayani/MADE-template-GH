import os
import pandas as pd
import sqlite3
import requests
from zipfile import ZipFile
from io import BytesIO

dataset_url_1 = "https://storage.googleapis.com/kaggle-data-sets/499/486063/compressed/seaice.csv.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240523%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240523T175451Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=36b8d6e5b4cf82bffb9f794872ec0ecf30dcdb610e4c288f5adb919fd369ed48d1b57e365d48d190caba258086d385562d33acc34c53239e1595693b86e702e6ca481a912ad93bd910b9dcd9286488081e112094cad703f1b599623580278d8ce9fb3826dac6abfb4dbce3896f303f64599786c32432576126f086a803518f3a9232c27e599873254161e214e892953c060d54c92bb05ed4b90d60bd702602fb3ae0d049a94d99db7ac33c98d6ef442d96b15a5f4f9270ba7231e1fa591cc6e398f1cb3a931b82b18d3d14f5fc3cb72c24d9f854f700b887aea94bc324c4b6acf85821e313b7ec25e77c00536abb7a06ba6c7b5a22cae75b7b657924a8ecb0db"
dataset_url_2 = "https://storage.googleapis.com/kaggle-data-sets/1056827/3028787/compressed/Environment_Temperature_change_E_All_Data_NOFLAG.csv.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240523%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240523T174202Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=751f1b5d5e8f0477a2815b699b260bf47fcdf87d78b05495faa61244001719cd3fa80becfba5bf48a1c5276832461cf0129eaf5c84a13288df83be90e494c7fd52a6c14bd1072cbd7090c61d72a414c5c3e172ed7bf75f6a643d60dfd4d0122dd7a9e93fca0f673baf6ee0ea1ed521d9d3fddc9b3c290fc5e44cb70c3fa6afe81f8c1d4792a7d6bc006549842710623ac5902b71de42d1d7a4a9e1f8b6cb984c62c9bea75a332322823b7dfa4350e493cf4b37f640f727af9b07199af5480e8dd0bb7bb1392eb8da8fcd711639fb56f6cd0a58ea953bbcc1c3d74e0611ffb30cb61b5e9ed9727974e19c6e17e0d9c67c87d0bdd65932ad56ee45d2a890b021f0"
data_path = "../data"
os.makedirs(data_path, exist_ok=True)


def download_and_extract_zip(url, extract_to='.'):
    response = requests.get(url)
    with ZipFile(BytesIO(response.content)) as thezip:
        thezip.extractall(path=extract_to)


def pipeline(urls, db_name, table_names):
    zip_path1 = os.path.join(data_path, 'dataset1.zip')
    zip_path2 = os.path.join(data_path, 'dataset2.zip')

    download_and_extract_zip(urls[0], data_path)
    download_and_extract_zip(urls[1], data_path)

    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    if len(csv_files) < 2:
        raise ValueError("Not enough CSV files.")

    df1 = pd.read_csv(os.path.join(data_path, csv_files[0]), encoding='cp1252')
    df2 = pd.read_csv(os.path.join(data_path, csv_files[1]))

    # Store DataFrames into SQLite database
    db_path = os.path.join(data_path, db_name)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    df1.to_sql(table_names[0], conn, if_exists="replace", index=False)
    df2.to_sql(table_names[1], conn, if_exists="replace", index=False)
    conn.close()


if __name__ == "__main__":
    pipeline([dataset_url_1, dataset_url_2], "iceTemp.db", ["ice", "temperature"])

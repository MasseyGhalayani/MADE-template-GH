import os
import pandas as pd
import sqlite3

dataset_url_1 = "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/5ba56a64-44fc-42ad-92ea-9cb34550e09c"
dataset_url_2 = "https://ieee-dataport.s3.amazonaws.com/open/18722/heart_statlog_cleveland_hungary_final.csv?response-content-disposition=attachment%3B%20filename%3D%22heart_statlog_cleveland_hungary_final.csv%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20240522%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240522T213841Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=c7f3b7a060484c65039fb5180beb24817af954f5b56cdbe2e939d3f2930f2883"
data_path = "../data"
os.makedirs(data_path, exist_ok=True)


def pipeline(urls, db_name, table_names):
    df1 = pd.read_csv(urls[0])
    df2 = pd.read_csv(urls[1])
    db_path = os.path.join(data_path, db_name)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    df1.to_sql(table_names[0], conn, if_exists="replace", index=False)
    df2.to_sql(table_names[1], conn, if_exists="replace", index=False)
    conn.close()


if __name__ == "__main__":
    pipeline([dataset_url_1,dataset_url_2], "heartData.db", ["heart", "heart_comprehensive"])

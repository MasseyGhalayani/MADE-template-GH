import os
import pandas as pd
import sqlite3

dataset_url = "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/5ba56a64-44fc-42ad-92ea-9cb34550e09c"
data_path = "../data"
os.makedirs(data_path, exist_ok=True)


def pipeline(url, db_name, table_name):
    df = pd.read_csv(url)
    db_path = os.path.join(data_path, db_name)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()


if __name__ == "__main__":
    pipeline(dataset_url, "heartData.db", "heart")

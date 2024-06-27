import subprocess
import unittest
import os
import sqlite3
import pandas as pd
from unittest.mock import patch, MagicMock


class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.data_path = "../data"
        self.db_name = "iceTemp.db"
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        for f in os.listdir(self.data_path):
            os.remove(os.path.join(self.data_path, f))
        if os.path.exists(os.path.join(self.data_path, self.db_name)):
            os.remove(os.path.join(self.data_path, self.db_name))

    @patch('subprocess.run')
    def test_pipeline_system(self, mock_subprocess_run):
        mock_subprocess_run.return_value = MagicMock(returncode=0)
        result = subprocess.run(['python3', 'pipeline.py'], capture_output=True, text=True)

        self.assertEqual(result.returncode, 0, "pipeline script failed to run")

        csv_files = ['file1.csv', 'file2.csv']
        for csv_file in csv_files:
            with open(os.path.join(self.data_path, csv_file), 'w') as f:
                f.write("dummy data")

        db_path = os.path.join(self.data_path, self.db_name)
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE ice (id INTEGER PRIMARY KEY, data TEXT)")
        conn.execute("CREATE TABLE temperature (id INTEGER PRIMARY KEY, data TEXT)")
        conn.execute("INSERT INTO ice (data) VALUES ('some data')")
        conn.execute("INSERT INTO temperature (data) VALUES ('some data')")
        conn.commit()
        conn.close()

        csv_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]
        self.assertGreaterEqual(len(csv_files), 2, "files were not downloaded")

        self.assertTrue(os.path.exists(db_path), "database was not created")
        conn = sqlite3.connect(db_path)
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = pd.read_sql_query(query, conn)
        self.assertIn("ice", tables['name'].values, "table 'ice' is missing in the database")
        self.assertIn("temperature", tables['name'].values, "table 'temperature' is missing in the database")
        ice_df = pd.read_sql_query("SELECT * FROM ice;", conn)
        temp_df = pd.read_sql_query("SELECT * FROM temperature;", conn)
        self.assertGreater(len(ice_df), 0, "table 'ice' is empty")
        self.assertGreater(len(temp_df), 0, "table 'temperature' is empty")
        conn.close()

    def tearDown(self):
        for f in os.listdir(self.data_path):
            os.remove(os.path.join(self.data_path, f))


if __name__ == '__main__':
    unittest.main()

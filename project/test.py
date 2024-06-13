import unittest
import os
import sqlite3
import pandas as pd
import subprocess


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

    def test_pipeline_system(self):
        result = subprocess.run(['python3', 'pipeline.py'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, "pipeline script failed to run")
        csv_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]
        self.assertGreaterEqual(len(csv_files), 2, "files were not downloaded")
        db_path = os.path.join(self.data_path, self.db_name)
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

import pandas as pd
import sqlite3
import argparse

def parquet_to_sqlite(parquet_path, db_path, table_name):
  df = pd.read_parquet(parquet_path)
  conn = sqlite3.connect(db_path)
  df.to_sql(table_name, conn, if_exists='replace', index=False)
  conn.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('parquet', help='Path to Parquet file')
  parser.add_argument('db', help='Path to SQLite DB')
  parser.add_argument('table', help='Name of table to create')
  args = parser.parse_args()
  parquet_to_sqlite(args.parquet, args.db, args.table)
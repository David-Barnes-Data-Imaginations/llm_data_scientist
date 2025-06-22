
"""
Utility module for creating SQLite tables from Parquet files.

This module provides functionality to read Turtle Graphics review and sales data
from Parquet files and store them in an SQLite database for easier querying and analysis.
"""

from pandas import DataFrame
from pathlib import Path
from typing import Final
import pandas as pd
import sqlite3

DATA_DIR: Final = Path("../../data")
REVIEWS_CSV: Final = DATA_DIR / "turtle_reviews.csv"
SALES_CSV: Final = DATA_DIR / "turtle_sales.csv"
REVIEWS_PARQUET: Final = DATA_DIR / "turtle_reviews.parquet"
SALES_PARQUET: Final = DATA_DIR / "turtle_sales.parquet"
DB_FILE: Final = DATA_DIR / "tg_database.db"


def create_table_from_parquet() -> None:
    """
    Create SQLite tables from Parquet files.

    Reads review and sales data from Parquet files and creates corresponding
    tables in an SQLite database. The function ensures the data directory exists,
    reads the Parquet files, and then writes the data to SQLite tables.

    Returns:
        None
    """
    # Ensure data directory exists
    DATA_DIR.mkdir(exist_ok=True)

    # Read Parquet files into DataFrames
    df_reviews: DataFrame = pd.read_parquet(REVIEWS_PARQUET)
    df_sales: DataFrame = pd.read_parquet(SALES_PARQUET)

    # Connect to (or create) SQLite DB
    conn = sqlite3.connect(DB_FILE)

    # Push DataFrame to SQLite
    df_reviews.to_sql('tg_reviews_table', conn, if_exists='replace', index=False)
    df_sales.to_sql('tg_sales_table', conn, if_exists='replace', index=False)

    # Finish
    conn.close()

if __name__ == "__main__":
    create_table_from_parquet()
from .prompts import SYSTEM_PROMPT
from .create_db_tables import create_table_from_parquet
from .file_reader import read_csv_summary, read_parquet_summary
from .generate_parquet import convert_to_parquet

# Import parquet_to_sqlite function using importlib
import importlib
parquet_to_sqlite_module = importlib.import_module("src.utils.parquet _to_sqlite")
parquet_to_sqlite = parquet_to_sqlite_module.parquet_to_sqlite

__all__ = [
    'SYSTEM_PROMPT',
    'create_table_from_parquet',
    'read_csv_summary',
    'read_parquet_summary',
    'convert_to_parquet',
    'parquet_to_sqlite',
]

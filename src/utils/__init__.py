"""
Utilities module for the data scientist project.

This module provides various utility functions for working with data files,
database operations, and system prompts.
"""

from .prompts import TCA_SYSTEM_PROMPT, TCA_MAIN_PROMPT, CHAT_PROMPT
from .create_db_tables import create_table_from_parquet
from .file_reader import read_csv_summary, read_parquet_summary
from .generate_parquet import convert_to_parquet
from .metadata_embedder import MetadataEmbedder
from .vllm_utils import check_vllm_server, wait_for_vllm_server, start_vllm_server_background

# Import parquet_to_sqlite function using relative import to handle the space in filename
# Note: Consider renaming the file to remove the space for better maintainability
from . import parquet_to_sqlite as parquet_to_sqlite_module
parquet_to_sqlite = parquet_to_sqlite_module.parquet_to_sqlite

__all__ = [
    'TCA_SYSTEM_PROMPT',
    'TCA_MAIN_PROMPT',
    'CHAT_PROMPT',
    'create_table_from_parquet',
    'read_csv_summary',
    'read_parquet_summary',
    'convert_to_parquet',
    'parquet_to_sqlite',
]

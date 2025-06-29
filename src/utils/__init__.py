"""
Utilities module for the data scientist project.

This module provides various utility functions for working with data files,
database operations, and system prompts.
"""
import os
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

from .prompts import TCA_SYSTEM_PROMPT, TCA_MAIN_PROMPT, CHAT_PROMPT
from .create_db_tables import create_table_from_parquet
from .file_reader import read_csv_summary, read_parquet_summary
from .generate_parquet import convert_to_parquet
from .metadata_embedder import MetadataEmbedder
from .ollama_utils import (
    check_ollama_server,
    wait_for_ollama_server,
    start_ollama_server_background,
    get_available_models,
    pull_model,
    generate_completion,
    chat_completion
)


__all__ = [
    'TCA_SYSTEM_PROMPT',
    'TCA_MAIN_PROMPT',
    'CHAT_PROMPT',
    'create_table_from_parquet',
    'read_csv_summary',
    'read_parquet_summary',
]

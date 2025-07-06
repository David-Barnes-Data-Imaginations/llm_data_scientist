"""
Utilities module for the data scientist project.

This module provides various utility functions for working with data files,
database operations, and system prompts.
"""
import os
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

from src.utils.prompts import CA_SYSTEM_PROMPT, CA_MAIN_PROMPT, CHAT_PROMPT
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
    'CA_SYSTEM_PROMPT',
    'CA_MAIN_PROMPT',
    'CHAT_PROMPT',
    'MetadataEmbedder'
]

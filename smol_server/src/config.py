from pathlib import Path
from dataclasses import dataclass
from openai import OpenAI
from src.tools.analysis_tools import *

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

# Server configuration
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8005
SERVER_ENDPOINT = f"https://{SERVER_HOST}:{SERVER_PORT}/sse"

# Model configuration
DEFAULT_MODEL = "Qwen2.5-Coder-32B.gguf"

@dataclass
class LLMConfig:
    tools: list = [dget_db_connection, query_sales, query_reviews, check_dataframe,
                             inspect_dataframe, analyze_data_patterns, document_learning_insights,
                             embed_and_store, retrieve_similar_chunks, validate_cleaning_results, save_cleaned_dataframe,
                             one_hot_encode, apply_feature_hashing, calculate_sparsity, handle_missing_values]
    model_name: str = "Qwen2.5-Coder-32B-Q4_K_L.gguf"
    max_tokens: int = 1024
    temperature: float = 0.2
    system_prompt: str = 'SYSTEM_PROMPT'

class Settings:
    llm_config = LLMConfig()
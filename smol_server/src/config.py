from pathlib import Path
from dataclasses import dataclass
from smolagents.local_python_executor import LocalPythonExecutor
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
    model_name: str = "Qwen2.5-Coder-32B-Q4_K_L.gguf"
    max_tokens: int = 1024
    temperature: float = 0.2
  #  system_prompt: str = 'SYSTEM_PROMPT'
    add_base_tools: bool = True
    prompt_templates: str = 'SYSTEM_PROMPT'
    # Set up a custom executor
    custom_executor = LocalPythonExecutor(
        ['sqlalchemy', 'random', 'sklearn', 'statistics', 'pandas',
         'itertools', 'queue', 'math'],
    )

class Settings:
    llm_config = LLMConfig()
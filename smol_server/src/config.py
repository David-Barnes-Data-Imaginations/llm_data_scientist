from pathlib import Path
from dataclasses import dataclass

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

# Server configuration
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8005
SERVER_ENDPOINT = f"https://{SERVER_HOST}:{SERVER_PORT}/sse"

# Model configuration
DEFAULT_MODEL = "Qwen2.5-Coder-32B-Q4_K_L.gguf"

@dataclass
class LLMConfig:
    model_name: str = "Qwen2.5-Coder-32B-Q4_K_L.gguf"
    max_tokens: int = 1024
    temperature: float = 0.7
    system_prompt: str = """You are an AI assistant specialized in data analysis and cleaning."""

class Settings:
    llm_config = LLMConfig()
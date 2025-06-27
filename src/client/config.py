from pathlib import Path
from dataclasses import dataclass
import os

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

@dataclass
class VLLMConfig:
    """Configuration for vLLM server"""
    host: str = "localhost"
    port: int = 8050  # Match your bash script
    model_name: str = "Qwen2.5-Coder-32B"
    api_base: str = "http://localhost:8050/v1"  # Match port
    api_key: str = "dummy-key"
    max_tokens: int = 8192
    temperature: float = 0.2
    top_p: float = 0.8
    stream: bool = True

@dataclass
class AgentConfig:
    """Configuration for the agent"""
    max_steps: int = 30
    planning_interval: int = 4
    verbosity_level: int = 2
    executor_type: str = "e2b"
    add_base_tools: bool = True

class Settings:
    vllm_config = VLLMConfig()
    agent_config = AgentConfig()
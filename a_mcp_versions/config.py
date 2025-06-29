from pathlib import Path
from dataclasses import dataclass
import os

from src.utils import CHAT_PROMPT, TCA_MAIN_PROMPT, TCA_SYSTEM_PROMPT

# Project root directory
ROOT_DIR = Path(__file__).parent.parent
templates = {
    "system": TCA_SYSTEM_PROMPT,
    "main": TCA_MAIN_PROMPT,
    "chat": CHAT_PROMPT
}

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


@dataclass
class AgentConfig:
    """Configuration for the agent"""
    max_steps: int = 30
    planning_interval: int = 4
    verbosity_level: int = 2
    executor_type: str = "e2b"
    add_base_tools: bool = True
    stream: bool = True
    prompt_templates: dict = templates

class Settings:
    vllm_config = VLLMConfig()
    agent_config = AgentConfig()
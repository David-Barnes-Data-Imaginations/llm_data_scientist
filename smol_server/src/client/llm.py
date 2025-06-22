from llama_index.llms import HuggingFaceInference
from llama_index.core import Settings
import os
from dotenv import load_dotenv

load_dotenv()

def setup_llm():
    """Initialize and configure the LLM."""
    api_token = os.getenv('HF_API_TOKEN')
    if not api_token:
        raise ValueError("HF_API_TOKEN environment variable is not set")
    
    llm = HuggingFaceInference(
        model_name="Qwen2.5-Coder-32B-Q4_K_L.gguf",
        token=api_token,
        max_new_tokens=512,
        temperature=0.7
    )
    Settings.llm = llm
    return llm
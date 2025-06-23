from langchain.llms import Ollama
from smolagents import LLMInterface
from src.config import Settings
# api_key = os.getenv("HF_API_TOKEN")
class OllamaWrapper(LLMInterface):
    """Wrapper to make Ollama work with CodeAgent"""
    def __init__(self):
        self.ollama = Ollama(
            model=Settings.llm_config.model_name,
            max_new_tokens=Settings.llm_config.max_tokens,
            temperature=Settings.llm_config.temperature,
        )

    async def generate(self, prompt: str) -> str:
        """Generate response using Ollama"""
        return await self.ollama.agenerate([prompt])

def setup_llm() -> OllamaWrapper:
    """Initialize and configure the LLM."""
    return OllamaWrapper()
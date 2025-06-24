from langchain_community.llms import VLLM
from smolagents import load_tool, CodeAgent
from src.config import Settings

# api_key = os.getenv("HF_API_TOKEN")
class VLLMWrapper(LLMInterface):
    """Wrapper to make vLLM work with CodeAgent"""
    def __init__(self):
        self.agent = CodeAgent(
            tools=Settings.llm_config.tools,
            model=Settings.llm_config.model_name,
            max_new_tokens=Settings.llm_config.max_tokens,
            temperature=Settings.llm_config.temperature,
            top_p=Settings.llm_config.top_p,
            planning_interval=Settings.llm_config.planning_interval,
            extra_body={
                Settings.llm_config.extra_body_key: Settings.llm_config.extra_body_value,
            },
        )

    async def generate(self, prompt: str) -> str:
        """Generate response using Ollama"""
        return await self.ollama.agenerate([prompt])

def setup_llm() -> OllamaWrapper:
    """Initialize and configure the LLM."""
    return OllamaWrapper()
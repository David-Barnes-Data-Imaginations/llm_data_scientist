from langchain_community.llms import VLLM
from smolagents import load_tool, CodeAgent
from src.config import Settings

# api_key = os.getenv("HF_API_TOKEN")
class VLLMWrapper(LLMInterface):
    """Initialize the CodeAgent with the specified model parameters from config.py and load tools from tools.py."""
    def __init__(self):
        self.agent = CodeAgent(
            tools=Settings.llm_config.tools,
            model=Settings.llm_config.model_name,
            max_new_tokens=Settings.llm_config.max_tokens,
            temperature=Settings.llm_config.temperature,
            top_p=Settings.llm_config.top_p,
            planning_interval=Settings.llm_config.planning_interval,
            add_base_tools=Settings.llm_config.add_base_tools,
            custom_executor=Settings.llm_config.custom_executor,
            additional_authorized_imports=Settings.llm_config.additional_authorized_imports,
            extra_body={
                Settings.llm_config.extra_body_key: Settings.llm_config.extra_body_value,
            },
        )
# **** Snippet to review from HF docs ***
    """
    def initialize_agent(model):

        return CodeAgent(
            tools=[WebSearchTool(), go_back, close_popups, search_item_ctrl_f],
            model=model,
            additional_authorized_imports=["helium"],
            step_callbacks=[save_screenshot],
            max_steps=20,
            verbosity_level=2,
        )
        """
# ******
    async def generate(self, prompt: str) -> str:
        """Generate response using Ollama"""
        return await self.ollama.agenerate([prompt])

def setup_llm() -> VLLMWrapper:
    """Initialize and configure the LLM."""
    return OllamaWrapper()
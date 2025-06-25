from langchain_community.llms import VLLM
from smolagents import ToolCallingAgent as BaseAgent
from docs_for_reference.smolagents_lib__tools.agents import ToolCallingAgent
from src.config import Settings
# api_key = os.getenv("HF_API_TOKEN")

# The LLM config file separator for the smolagents configuration settings
# All settings added whilst I understand what they all do

class VLLMWrapper(BaseAgent):
    """Initialize the CodeAgent with the specified model parameters from config.py and load tools from tools.py."""
    def __init__(self):
     #  self.agent = CodeAgent( # Change if the TCA doesn't cause singularity
        self.agent = ToolCallingAgent(
         #   tools=Settings.llm_config.tools,  #
            model=Settings.llm_config.model_name,
            max_completion_tokens=Settings.llm_config.max_completion_tokenss,
            temperature=Settings.llm_config.temperature,
            top_p=Settings.llm_config.top_p,
            planning_interval=Settings.llm_config.planning_interval,
            add_base_tools=Settings.llm_config.add_base_tools,
            executor_type=Settings.llm_config.custom_executor,
            additional_authorized_imports=Settings.llm_config.additional_authorized_imports,
            verbosity_level=2,
            stream_outputs=True,
            step_callbacks=[],
            extra_body={
                Settings.llm_config.extra_body_key: Settings.llm_config.extra_body_value,
            },
        )

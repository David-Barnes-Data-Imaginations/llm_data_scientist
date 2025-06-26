
from smolagents import ToolCallingAgent as BaseAgent
from src.client.config import Settings
from smolagents import ToolCallingAgent as BaseAgent

"""Initialize the Agent with the specified model parameters"""
class CustomAgent(BaseAgent):
        super().__init__(
                #   tools=Settings.llm_config.tools,  #
                model=Settings.model_name,
                max_completion_tokens=Settings.max_completion_tokens,
                temperature=Settings.temperature,
                top_p=Settings.top_p,
                top_k=Settings.top_k,
                system_prompt=Settings.system_prompt,
                planning_interval=Settings.planning_interval,
                add_base_tools=Settings.add_base_tools,
                executor_type=Settings.custom_executor,
#               additional_authorized_imports=Settings.additional_authorized_imports,
                verbosity_level=Settings.verbosity_level,
                stream_outputs=Settings.stream_outputs
        )




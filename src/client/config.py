from pathlib import Path
from dataclasses import dataclass
from smolagents import CodeAgent
from openai import OpenAI
from src.tools.analysis_tools import *
from src.utils.prompts import SYSTEM_PROMPT, MAIN_PROMPT, CHAT_PROMPT
# Project root directory
ROOT_DIR = Path(__file__).parent.parent

# Server configuration
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8005
SERVER_ENDPOINT = f"https://{SERVER_HOST}:{SERVER_PORT}/stdio"

templates = {SYSTEM_PROMPT: SYSTEM_PROMPT, MAIN_PROMPT: MAIN_PROMPT, CHAT_PROMPT: CHAT_PROMPT}

# Model configuration
DEFAULT_MODEL = "Qwen2.5-Coder-32B.gguf"

@dataclass
class LLMConfig:
    class LLMConfig:
        """
        Agent class that solves the given task step by step, using the ReAct framework:
        While the objective is not reached, the agent will perform a cycle of action (given by the LLM) and observation (obtained from the environment).

        Args:
            tools (`list[Tool]`): [`Tool`]s that the agent can use.
            model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
            prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
            instructions (`str`, *optional*): Custom instructions for the agent, will be inserted in the system prompt.
            max_steps (`int`, default `20`): Maximum number of steps the agent can take to solve the task.
            add_base_tools (`bool`, default `False`): Whether to add the base tools to the agent's tools.
            verbosity_level (`LogLevel`, default `LogLevel.INFO`): Level of verbosity of the agent's logs.
            grammar (`dict[str, str]`, *optional*): Grammar used to parse the LLM output.
                <Deprecated version="1.17.0">
                Parameter `grammar` is deprecated and will be removed in version 1.20.
                </Deprecated>
            managed_agents (`list`, *optional*): Managed agents that the agent can call.
            step_callbacks (`list[Callable]`, *optional*): Callbacks that will be called at each step.
            planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
            name (`str`, *optional*): Necessary for a managed agent only - the name by which this agent can be called.
            description (`str`, *optional*): Necessary for a managed agent only - the description of this agent.
            provide_run_summary (`bool`, *optional*): Whether to provide a run summary when called as a managed agent.
            final_answer_checks (`list[Callable]`, *optional*): List of validation functions to run before accepting a final answer.
                Each function should:
                - Take the final answer and the agent's memory as arguments.
                - Return a boolean indicating whether the final answer is valid.
        """

    model_name: str = DEFAULT_MODEL
    max_completion_tokens: int = 8192,
    temperature: float = 0.2
    top_p: float = 0.8
    top_k: int = 20
    prompt_templates: dict = templates,
    add_base_tools: bool = True
    executor_type="e2b" # The custom executor for E2B
    max_steps=30 # arbitrary until I see it run
    planning_interval=4 # Interval at which agents runs planning steps
    # add one at a time once CodeAgent running
    # additional_authorized_imports=['sqlalchemy', 'random', 'sklearn', 'statistics', 'pandas', 'itertools', 'queue', 'math']

class Settings:
    llm_config = LLMConfig()
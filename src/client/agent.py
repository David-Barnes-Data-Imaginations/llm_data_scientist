from smolagents import load_tool
from smolagents import ToolCallingAgent
from typing import List
from smolagents import Tool

from src.utils import TCA_SYSTEM_PROMPT, TCA_MAIN_PROMPT, CHAT_PROMPT

templates = {TCA_SYSTEM_PROMPT, TCA_MAIN_PROMPT, CHAT_PROMPT}

"""Initialize the Agent with the specified model parameters"""
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
# All tools (custom + HF Hub) go in the same parameter
agent = ToolCallingAgent(
        tools = [],
        model_name = "Qwen2.5-Coder-32B.gguf",
        max_completion_tokens = 8192,
        temperature = 0.2,
        top_p = 0.8,
        top_k = 20,
        prompt_templates = templates,
        add_base_tools = True,
        executor_type="e2b", # The custom executor for E2B
        max_steps=30, # arbitrary until I see it run
        planning_interval=4, # Interval at which agents runs planning steps
        verbosity_level=2,
        stream_outputs=True,
    # add one at a time once CodeAgent running
                # additional_authorized_imports=['sqlalchemy', 'random', 'sklearn', 'statistics', 'pandas', 'itertools', 'queue', 'math']
                )

class ToolFactory:
        def __init__(self, sandbox):
                self.sandbox = sandbox

            def create_all_tools(self) -> List[Tool]:

            # Import your custom tools
            from tools.data_structure_inspection_tools import InspectDataframe, ValidateData, DataValidator, AnalyzePatterns, CheckDataframe
            from tools.database_tools import DatabaseConnect, QuerySales, QueryReviews
            from tools.documentation_tools import DocumentLearningInsights, EmbedAndStore, RetrieveSimilarChunks, ValidateCleaningResults, RetrieveMetadata
            from tools.data_structure_feature_engineering_tools import OneHotEncode, ApplyFeatureHashing, CalculateSparsity, HandleMissingValues
            from tools.dataframe_tools import DataframeConcat, DataframeDrop, DataframeMerge, DataframeFill, DataframeMelt, DataframeToNumeric

            # Create instances of your custom tools
            all_tools =[
                DatabaseConnect(sandbox=self.sandbox),
                QuerySales(sandbox=self.sandbox),
                QueryReviews(sandbox=self.sandbox),
                InspectDataframe(sandbox=self.sandbox),
                ValidateData(sandbox=self.sandbox),
                DataValidator(sandbox=self.sandbox),
                AnalyzePatterns(sandbox=self.sandbox),
                CheckDataframe(sandbox=self.sandbox),
                DocumentLearningInsights(sandbox=self.sandbox),
                EmbedAndStore(sandbox=self.sandbox),
                RetrieveSimilarChunks(sandbox=self.sandbox),
                ValidateCleaningResults(sandbox=self.sandbox),
                RetrieveMetadata(sandbox=self.sandbox),
                OneHotEncode(sandbox=self.sandbox),
                ApplyFeatureHashing(sandbox=self.sandbox),
                CalculateSparsity(sandbox=self.sandbox),
                HandleMissingValues(sandbox=self.sandbox),
                DataframeConcat(sandbox=self.sandbox),
                DataframeDrop(sandbox=self.sandbox),
                DataframeMerge(sandbox=self.sandbox),
                DataframeFill(sandbox=self.sandbox),
                DataframeMelt(sandbox=self.sandbox),
                DataframeToNumeric(sandbox=self.sandbox)
            ]
            return all_tools

            # Optionally add HF Hub tools
            # hf_tools = [
            #     load_tool("huggingface-tools/text-classification"),
            # ]


async def list_tools(tools):
        """List all available tools."""
        for tool in tools:
                print(f"Tool: {tool.name} - {getattr(tool, 'description', 'No description')}")
        return tools


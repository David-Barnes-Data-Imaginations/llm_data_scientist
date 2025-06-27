from smolagents import ToolCallingAgent
from smolagents.models import OpenAIModel
from typing import List
from smolagents import Tool
from src.utils.prompts import TCA_SYSTEM_PROMPT, TCA_MAIN_PROMPT, CHAT_PROMPT

class CustomAgent:
    """Custom agent wrapper that configures ToolCallingAgent with our tools and settings"""
    
    def __init__(self, tools: List[Tool] = None, sandbox=None, metadata_embedder=None):
        self.sandbox = sandbox
        self.metadata_embedder = metadata_embedder
        self.tools = tools or []
        
        # Prompt templates
        templates = {
            "system": TCA_SYSTEM_PROMPT,
            "main": TCA_MAIN_PROMPT, 
            "chat": CHAT_PROMPT
        }
        
        # Create OpenAI-compatible model pointing to vLLM server
        model = OpenAIModel(
            model_id="Qwen2.5-Coder-32B",  # This should match --served-model-name
            api_key="dummy-key",  # vLLM doesn't require real API key for local server
            api_base="http://localhost:8050/v1",  # vLLM server endpoint
            max_tokens=8192,
            temperature=0.2,
            top_p=0.8,
            # Note: top_k is handled by vLLM server, not passed here
        )
        
        # Create the actual ToolCallingAgent
        self.agent = ToolCallingAgent(
            tools=self.tools,
            model=model,  # Use the vLLM model instead of model_name
            prompt_templates=templates,
            add_base_tools=True,
            executor_type="e2b",
            max_steps=30,
            planning_interval=4,
            verbosity_level=2,
            stream_outputs=True,
        )
        
        # Optional telemetry (will be set in main.py)
        self.telemetry = None
    
    def run(self, task: str):
        """Run the agent on a task"""
        return self.agent.run(task)
    
    def chat(self, message: str):
        """Chat with the agent"""
        return self.agent.chat(message)
    
    def __getattr__(self, name):
        """Delegate any missing attributes to the underlying agent"""
        return getattr(self.agent, name)

# ... rest of ToolFactory stays the same
from typing import List
from smolagents import Tool
from src.utils.prompts import TCA_SYSTEM_PROMPT, TCA_MAIN_PROMPT, CHAT_PROMPT

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

class ToolFactory:
    """Factory for creating all tools with proper dependencies"""
    
    def __init__(self, sandbox, metadata_embedder=None):
        self.sandbox = sandbox
        self.metadata_embedder = metadata_embedder

    def create_all_tools(self) -> List[Tool]:
        """Create all tools with sandbox and metadata_embedder dependencies injected"""
        
        # Import your custom tools
        from tools.data_structure_inspection_tools import InspectDataframe, CheckDataframe, AnalyzePatterns
        from tools.database_tools import DatabaseConnect, QuerySales, QueryReviews
        from tools.documentation_tools import (DocumentLearningInsights, EmbedAndStore, 
                                               RetrieveMetadata, RetrieveSimilarChunks,
                                               ValidateCleaningResults, SaveCleanedDataframe)
        from tools.data_structure_feature_engineering_tools import (OneHotEncode, ApplyFeatureHashing,
                                                                    CalculateSparsity, HandleMissingValues)

        # Create instances of your custom tools
        tools = [
            DatabaseConnect(sandbox=self.sandbox),
            QuerySales(sandbox=self.sandbox),
            QueryReviews(sandbox=self.sandbox),
            CheckDataframe(sandbox=self.sandbox),
            InspectDataframe(sandbox=self.sandbox),
            AnalyzePatterns(sandbox=self.sandbox),
            DocumentLearningInsights(sandbox=self.sandbox),
            EmbedAndStore(sandbox=self.sandbox),
            RetrieveMetadata(sandbox=self.sandbox, metadata_embedder=self.metadata_embedder),
            RetrieveSimilarChunks(sandbox=self.sandbox),
            ValidateCleaningResults(sandbox=self.sandbox),
            SaveCleanedDataframe(sandbox=self.sandbox),
            OneHotEncode(sandbox=self.sandbox),
            ApplyFeatureHashing(sandbox=self.sandbox),
            CalculateSparsity(sandbox=self.sandbox),
            HandleMissingValues(sandbox=self.sandbox)
        ]
        
        return tools
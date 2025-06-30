from smolagents import ToolCallingAgent
from typing import List
from smolagents import Tool
from a_mcp_versions.prompts import TCA_SYSTEM_PROMPT, TCA_MAIN_PROMPT, CHAT_PROMPT

# Prompt templates
templates = {
    "system": TCA_SYSTEM_PROMPT,
    "main": TCA_MAIN_PROMPT,
    "chat": CHAT_PROMPT
}

class CustomAgent:
    """Custom agent wrapper that configures ToolCallingAgent with our tools and settings"""

    def __init__(self, tools: List[Tool] = None, sandbox=None, metadata_embedder=None, model_id=None):
        self.metadata_embedder = metadata_embedder
        self.tools = tools or []

        # Default to Ollama model if not specified
        if model_id is None:
            model_id = "ollama://DeepSeek-R1-Distill"  # or whatever model you have in Ollama

        # Create the ToolCallingAgent with Ollama model
        self.agent = ToolCallingAgent(
            tools=self.tools,
            model=model_id,  # Add this line to specify Ollama model
            system_prompt=templates["system"],
            prompt=templates["main"],
            chat_prompt=templates["chat"],
            add_base_tools=True,
            executor_type="e2b",
            max_steps=30,
            planning_interval=4,
            verbosity_level=2,
            stream_outputs=True,
        )

        self.telemetry = None

    def run(self, task: str):
        """Run the agent on a task"""
        return self.agent.run(task)

 #   def __getattr__(self, name):
        """Delegate any missing attributes to the underlying agent"""
 #       return getattr(self.agent, name)

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
        from tools.documentation_tools import (DocumentLearningInsights,
                                               RetrieveMetadata, RetrieveSimilarChunks,
                                               ValidateCleaningResults, SaveCleanedDataframe)
        from tools.data_structure_feature_engineering_tools import (OneHotEncode, ApplyFeatureHashing,
                                                                    CalculateSparsity, HandleMissingValues)

        # Create instances of your custom tools
        tools = [
            DatabaseConnect(sandbox=self.sandbox, ),
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
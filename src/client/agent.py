import datetime
import asyncio  # Add this
import json     # Add this
from smolagents import ToolCallingAgent
from typing import List
from smolagents import Tool
from src.utils.prompts import TCA_SYSTEM_PROMPT, TCA_MAIN_PROMPT, CHAT_PROMPT
T
# Prompt templates
templates = {
    "system": TCA_SYSTEM_PROMPT,
    "main": TCA_MAIN_PROMPT,
    "chat": CHAT_PROMPT
}

# StepController handles step management by adding a time delay, alongside the manual controls
class StepController:
    def __init__(self):
        self.ready = asyncio.Event()
        self.manual_mode = False

    def next(self):
        self.ready.set()

    async def wait(self):
        if self.manual_mode:
            await self.ready.wait()
            self.ready.clear()
        else:
            await asyncio.sleep(0.5)

    def toggle_mode(self, manual: bool):
        self.manual_mode = manual
        if not manual:
            self.ready.set()  # Unblock any pending waits


# Defines the Custom agent, which is currently a ToolCallingAgent. CodeAgent has same structure but switches prompt
class CustomAgent:
    """Custom agent wrapper that configures ToolCallingAgent with our tools and settings"""

    def __init__(self, tools: List[Tool] = None, sandbox=None, metadata_embedder=None, model_id=None):
        self.metadata_embedder = metadata_embedder
        self.tools = tools or []

        if model_id is None:
            model_id = "ollama://DeepSeek-R1-Distill"

        self.agent = ToolCallingAgent(
            tools=self.tools,
            model=model_id,
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
        self.controller = StepController()

    def run(self, task: str):
        return self.agent.run(task)

    def log_agent_step(self, thought: str, tool: str = "", params: dict = None, result: str = ""):
        event = {
            "thought": thought,
            "tool": tool,
            "params": params or {},
            "result": result
        }
        if self.telemetry:
            self.telemetry.log_agent_step(event)
        with open("states/agent_step_log.jsonl", "a") as f:
            f.write(json.dumps(event) + "\n")
        print("🧠 AGENT STEP saved to log.")
        return event

    async def agent_runner(self, task: str):
        trace = self.agent.run(task)
        for step in trace.steps:
            if hasattr(step, "tool_name"):
                self.log_agent_step(
                    thought=getattr(step, "thought", ""),
                    tool=step.tool_name,
                    params=step.tool_input,
                    result=step.observation
                )
                yield {
                    "thought": getattr(step, "thought", ""),
                    "tool_name": step.tool_name,
                    "tool_input": step.tool_input,
                    "observation": step.observation
                }
                await self.controller.wait()
            elif hasattr(step, "message"):
                yield {
                    "thought": step.message.content
                }
                await self.controller.wait()
            pass

    def toggle_manual_mode(self, manual: bool):
        self.controller.toggle_mode(manual)

    def next_step(self):
        self.controller.next()

class ToolFactory:
    """Factory for creating all tools with proper dependencies"""

    def __init__(self, sandbox, metadata_embedder=None):
        self.sandbox = sandbox
        self.metadata_embedder = metadata_embedder

    def create_all_tools(self) -> List[Tool]:
        """Create all tools with sandbox and metadata_embedder dependencies injected"""

        # Import your custom tools
        from tools.data_structure_inspection_tools import InspectDataframe, CheckDataframe, AnalyzePatterns, ValidateData
        from tools.database_tools import DatabaseConnect, QuerySales, QueryReviews
        from tools.documentation_tools import (DocumentLearningInsights,
                                               RetrieveMetadata, RetrieveSimilarChunks,
                                               ValidateCleaningResults, SaveCleanedDataframe, GetToolHelp)
        from tools.data_structure_feature_engineering_tools import CalculateSparsity, HandleMissingValues
        from tools.dataframe_manipulation_tools import DataframeMelt, DataframeConcat, DataframeDrop, DataframeFill, DataframeMerge, DataframeToNumeric
        from tools.dataframe_storage import CreateDataframe, CopyDataframe

        # Create instances of your custom tools
        tools = [
            DatabaseConnect(sandbox=self.sandbox, ),
            QuerySales(sandbox=self.sandbox),
            QueryReviews(sandbox=self.sandbox),
            ValidateData(sandbox=self.sandbox),
            CheckDataframe(sandbox=self.sandbox),
            InspectDataframe(sandbox=self.sandbox),
            AnalyzePatterns(sandbox=self.sandbox),
            DocumentLearningInsights(sandbox=self.sandbox),
            RetrieveMetadata(sandbox=self.sandbox, metadata_embedder=self.metadata_embedder),
            RetrieveSimilarChunks(sandbox=self.sandbox),
            ValidateCleaningResults(sandbox=self.sandbox),
            SaveCleanedDataframe(sandbox=self.sandbox),
            CalculateSparsity(sandbox=self.sandbox),
            HandleMissingValues(sandbox=self.sandbox),
            DataframeMelt(sandbox=self.sandbox),
            DataframeConcat(sandbox=self.sandbox),
            DataframeDrop(sandbox=self.sandbox),
            DataframeFill(sandbox=self.sandbox),
            DataframeMerge(sandbox=self.sandbox),
            DataframeToNumeric(sandbox=self.sandbox),
            CreateDataframe(sandbox=self.sandbox),
            CopyDataframe(sandbox=self.sandbox),
            GetToolHelp(sandbox=self.sandbox),
        ]
        return tools

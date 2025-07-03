import datetime
import asyncio  # Add this
import json     # Add this
from smolagents import CodeAgent, PromptTemplates
from smolagents.models import LiteLLMModel
from typing import List
from smolagents import Tool
from src.utils.prompts import CA_SYSTEM_PROMPT, TASK_PROMPT, PLANNING_INITIAL_FACTS, PLANNING_INITIAL_PLAN, \
    PLANNING_UPDATE_FACTS_PRE, PLANNING_UPDATE_FACTS_POST, PLANNING_UPDATE_PLAN_PRE, PLANNING_UPDATE_PLAN_POST, \
    CA_MAIN_PROMPT, TCA_MAIN_PROMPT
import litellm
from smolagents.agent_types import AgentText


from smolagents.local_python_executor import LocalPythonExecutor

# Prompt templates
prompt_templates = {
    "system_prompt": CA_SYSTEM_PROMPT,
    "planning": {
        "initial_facts": PLANNING_INITIAL_FACTS,
        "initial_plan": PLANNING_INITIAL_PLAN,
        "update_facts_pre_messages": PLANNING_UPDATE_FACTS_PRE,
        "update_facts_post_messages": PLANNING_UPDATE_FACTS_POST,
        "update_plan_pre_messages": PLANNING_UPDATE_PLAN_PRE,
        "update_plan_post_messages": PLANNING_UPDATE_PLAN_POST
    },
    "managed_agent": {
        "task": CA_MAIN_PROMPT,
        "report": ""
    }
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

    def __init__(self, tools: List[Tool] = None, sandbox=None, metadata_embedder=None, model_id=None, executor=None):
        self.metadata_embedder = metadata_embedder
        self.tools = tools or []

        try:
            test_response = litellm.completion(
                model="ollama/DeepSeek-R1",
                messages=[{"role": "user", "content": "Hello, i'm just joining now, i'll be 1 minute" }],
                api_base="http://localhost:11434",
                stream=False
            )
            print("✅ Ollama test response:", test_response['choices'][0]['message']['content'])
        except Exception as e:
            print("❌ Ollama sanity check failed:", str(e))

        # model_id=f"ollama_chat/{model_id}" — this is apparently incorrect for newer LiteLLM + Ollama
        # ✅ Fix:
        model_id = "ollama/DeepSeek-R1"  # ✅ Correct LiteLLM + Ollama model name

        model = LiteLLMModel(
            model_id=model_id,
            api_base="http://localhost:11434",  # still valid
            api_key="dummy",                   # fine for Ollama
            num_ctx=8192                        # fine to override context
        )


    # Optionally raise or fallback here
        # Use custom executor if provided, otherwise default to e2b
        if executor:
            self.agent = CodeAgent(
                tools=self.tools,
                model=model,
                # Remove custom prompt templates for now to use defaults
                # prompt_templates=prompt_templates,
                executor=executor,
                additional_authorized_imports=["pandas sqlalchemy sklearn statistics math "],
                use_structured_outputs_internally=True,
                add_base_tools=True,
                max_steps=30,
                verbosity_level=2,
            )
        else:
            self.agent = CodeAgent(
                tools=self.tools,
                model=model,
                executor_type="e2b",
                additional_authorized_imports=["pandas sqlalchemy "],
                use_structured_outputs_internally=True,
                add_base_tools=True,
                max_steps=30,
                verbosity_level=2,
            )

        self.telemetry = None
        self.controller = StepController()

    def run(self, task: str, images=None, stream=False, reset=False, additional_args=None):
        # Pass through to the underlying agent with proper streaming support
        if stream:
            return self.agent.run(task, stream=True)
        else:
            return self.agent.run(task)

    def cleanup(self):
        """Clean up agent resources including E2B sandbox."""
        try:
            if hasattr(self.agent, 'cleanup'):
                self.agent.cleanup()
                print("✅ Agent cleanup completed")
            else:
                print("ℹ️ Agent doesn't have cleanup method")
        except Exception as e:
            print(f"⚠️ Error during agent cleanup: {e}")

    def __enter__(self):
        """Support for context manager usage."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup when exiting context manager."""
        self.cleanup()

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
    reasoning_preface = AgentText(text="Can you reason through this step by step before taking any action?\n" + task)
    trace = self.agent.run(reasoning_preface)

    for i, step in enumerate(trace.steps):
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
        elif hasattr(step, "message"):
            yield {
                "thought": step.message.content
            }

        # Manual mode pause or 0.5s fallback
        await self.controller.wait()

        # Slight breathing room after first step
        if i == 0:
            await asyncio.sleep(1.2)

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
                                               ValidateCleaningResults, SaveCleanedDataframe)
        from tools.data_structure_feature_engineering_tools import CalculateSparsity, HandleMissingValues
        from tools.dataframe_manipulation_tools import DataframeMelt, DataframeConcat, DataframeDrop, DataframeFill, DataframeMerge, DataframeToNumeric
        from tools.dataframe_storage import CreateDataframe, CopyDataframe
        from tools.help_tools import GetToolHelp
        from tools.code_tools import RunCodeRaiseErrors, RunSQL
        from smolagents import (
            WebSearchTool,
            VisitWebpageTool,
        )

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
            GetToolHelp(sandbox=self.sandbox, metadata_embedder=self.metadata_embedder),
            RunCodeRaiseErrors(sandbox=self.sandbox),
            RunSQL(sandbox=self.sandbox),
            WebSearchTool(),
            VisitWebpageTool(),
        ]
        return tools

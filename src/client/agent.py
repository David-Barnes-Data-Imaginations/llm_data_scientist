import datetime
import asyncio  # Add this
import json     # Add this
from smolagents import CodeAgent, PromptTemplates
from smolagents.models import LiteLLMModel
from typing import List, Any, AsyncGenerator
from smolagents import Tool
from src.utils.prompts import CA_SYSTEM_PROMPT, PLANNING_INITIAL_FACTS, PLANNING_INITIAL_PLAN, \
    PLANNING_UPDATE_FACTS_PRE, PLANNING_UPDATE_FACTS_POST, PLANNING_UPDATE_PLAN_PRE, PLANNING_UPDATE_PLAN_POST, \
    CA_MAIN_PROMPT
import litellm
from typing import Generator
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
    },
        "final_answer": {
        "pre_messages": "",
        "post_messages": ""
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
            await asyncio.sleep(1.5)

    def toggle_mode(self, manual: bool):
        self.manual_mode = manual
        if not manual:
            self.ready.set()  # Unblock any pending waits

class CustomAgent:
    """Custom agent wrapper that configures ToolCallingAgent with our tools and settings"""

    def __init__(self, tools: List[Tool] = None, sandbox=None, metadata_embedder=None, model_id=None, executor=None):
        self.metadata_embedder = metadata_embedder
        self.tools = tools or []
        self.is_agentic_mode = False

        """Custom agent wrapper that configures ToolCallingAgent with our tools and settings"""

        if model_id is None:
            model_id = "gemma3:12b"

        # Ensure correct LiteLLM + Ollama model name format
        if not model_id.startswith("ollama/"):
            model_id = f"ollama/{model_id}"

        model = LiteLLMModel(
            model_id=model_id,
            api_base="http://localhost:11434",  # still valid
            api_key="dummy",                   # fine for Ollama
            num_ctx=8192                        # fine to override context
        )

        # Store sandbox reference for tools to use
        self.sandbox = sandbox
        
        self.agent = CodeAgent(
            tools=self.tools,
            model=model,
            prompt_templates=prompt_templates,
            additional_authorized_imports=["pandas sqlalchemy scikit-learn statistics smolagents seaborn"],
            executor_type="e2b",
            use_structured_outputs_internally=True,
            planning_interval=5,
            add_base_tools=True,  # Enable base tools including python_interpreter
            max_steps=30,
            verbosity_level=2,
        )
        
        # Copy files from our sandbox to the agent's sandbox
        self._setup_agent_data()

        try:
            test_response = litellm.completion(
                model="ollama/qwen2.5-coder:32b",
                messages=[{"role": "user", "content": "." }],
                api_base="http://localhost:11434",
                stream=False
            )
            print("✅ Ollama test response:", test_response['choices'][0]['message']['content'])
        except Exception as e:
            print("❌ Ollama sanity check failed:", str(e))


    def _setup_agent_data(self):
        """Copy data files from main sandbox to agent's sandbox"""
        try:
            # Get the agent's executor (sandbox)
            agent_sandbox = self.agent.python_executor.sandbox
            
            # Copy the data files from our sandbox to the agent's sandbox
            
            # Copy turtle_reviews.csv
            turtle_reviews_content = self.sandbox.files.read("/data/turtle_reviews.csv")
            agent_sandbox.files.write("/data/turtle_reviews.csv", turtle_reviews_content)
            
            # Copy turtle_sales.csv  
            turtle_sales_content = self.sandbox.files.read("/data/turtle_sales.csv")
            agent_sandbox.files.write("/data/turtle_sales.csv", turtle_sales_content)
            
            # Copy database file
            db_content = self.sandbox.files.read("/data/tg_database.db")
            agent_sandbox.files.write("/data/tg_database.db", db_content)
            
            # Copy metadata file
            metadata_content = self.sandbox.files.read("/data/metadata/turtle_games_dataset_metadata.md")
            agent_sandbox.files.write("/data/metadata/turtle_games_dataset_metadata.md", metadata_content)
            
            print("✅ Successfully copied data files to agent's sandbox")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not copy files to agent sandbox: {str(e)}")

    def run(self, task: str, images=None, stream=False, reset=False, additional_args=None):
        print(f"🔍 Agent.run called with task: '{task}', agentic_mode: {self.is_agentic_mode}")
        
        # Check if user wants to begin agentic workflow
        if task.lower().strip() == "begin":
            self.is_agentic_mode = True
            print("🚀 Switching to agentic mode")
            return self.start_agentic_workflow()
        
        # If in agentic mode, handle differently
        if self.is_agentic_mode:
            print("🤖 Handling in agentic mode")
            return self.handle_agentic_mode(task, images, stream, reset, additional_args)
        
        # Otherwise, run in normal chat mode
        print("💬 Handling in chat mode")
        return self.handle_chat_mode(task, images, stream, reset, additional_args)

    def handle_chat_mode(self, task: str, images=None, stream=False, reset=False, additional_args=None):
        """Handle normal chat interactions"""
        # Use the same model as the agent for consistency
        try:
            # Get the model_id from the agent's model
            model_id = self.agent.model.model_id
            
            response = litellm.completion(
                model=model_id,
                messages=[{"role": "user", "content": task}],
                api_base="http://localhost:11434",
                stream=stream
            )
            
            if stream:
                return response
            else:
                return response['choices'][0]['message']['content']
        except Exception as e:
            return f"Error in chat mode: {str(e)}"

    def handle_agentic_mode(self, task: str, images=None, stream=False, reset=False, additional_args=None):
        """Handle agentic workflow execution"""
        # Pass through to the underlying agent with proper streaming support
        if stream:
            return self.agent.run(task, stream=True)
        else:
            return self.agent.run(task)

    def start_agentic_workflow(self):
        """Start the agentic workflow"""
        return """🚀 Starting agentic workflow! I'm now in analysis mode. 

I can help you clean and analyze this data using a systematic, chunk-based approach.

I have access to the Turtle Games dataset with the following files:
- Reviews CSV: /data/turtle_reviews.csv

I should clean the dataset in chunks of 200 rows and use the SaveCleanedDataframe tool to save the results.
Once all chunks are cleaned, I will print the dataframe and submit "Dataframe cleaned" as my final answer.

To run code, use the 'RunCode' tool without any ticks:
Correct: RunCode(import pandas as pd)
Incorrect: RunCode(```python import pandas as pd```)


 """

    def return_to_chat_mode(self):
        """Return to chat mode after agentic workflow completes"""
        self.is_agentic_mode = False
        return "✅ Analysis complete! I'm back in chat mode. Feel free to ask me questions about the analysis or request new tasks."

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

# --- Context Manager ---
class SmartContextManager:
    def __init__(self, limit_tokens=80000):
        self.history = []
        self.limit_tokens = limit_tokens

    def _approx_tokens(self, text: str) -> int:
        return int(len(text.split()) * 0.75)

    def add(self, message: str):
        self.history.append(message)
        while self._approx_tokens("\n".join(self.history)) > self.limit_tokens:
            self.history.pop(0)

    def get(self) -> str:
        return "\n".join(self.history)


    async def agent_runner(self, task: str) -> AsyncGenerator[dict[str, Any] | dict[str, str], None]:
        """Run agent with context management and return to chat mode after final_answer"""
        reasoning_preface = AgentText(text=f"Can you reason through this step by step before taking any action?\n{task}")
        self.context = SmartContextManager()
        trace = self.agent.run(reasoning_preface)
        self.controller = StepController()

        for i, step in enumerate(trace.steps):
            # Append the step prompt to context
            if hasattr(step, "message"):
                self.context.add(f"Step {i}: {step.message.content}")

            elif hasattr(step, "thought") and hasattr(step, "tool_name"):
                # Log the reasoning chain to context and store insight in RAG
                insight = f"Thought: {step.thought}\nTool: {step.tool_name}\nParams: {step.tool_input}\nObservation: {step.observation}"
                self.context.add(insight)

                # Store insight in RAG if supported
                if hasattr(self.agent, "store_insight"):
                    self.agent.store_insight(insight)

                yield {
                    "thought": step.thought,
                    "tool_name": step.tool_name,
                    "tool_input": step.tool_input,
                    "observation": step.observation
                }

            else:
                yield {
                    "step_info": str(step)
                }

            # Manual mode pause or 0.5s fallback
            await self.controller.wait(2 if self.controller.manual_mode else 2)

            # Slight breathing room after first step
            if i == 0:
                await asyncio.sleep(0.5)

        # Check if we have a final answer and signal return to chat mode
        if hasattr(trace, 'final_answer') and trace.final_answer:
            yield {
                "final_answer": str(trace.final_answer),
                "return_to_chat": True
            }

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

        # Import your custom tool
        from tools.documentation_tools import (DocumentLearningInsights,
                                               RetrieveMetadata, RetrieveSimilarChunks,
                                               ValidateCleaningResults)
        from tools.dataframe_storage import SaveCleanedDataframe
        from tools.help_tools import GetToolHelp
        from tools.code_tools import RunCode
        

        # Create instances of your custom tools
        tools = [
            DocumentLearningInsights(sandbox=self.sandbox),
            RetrieveMetadata(sandbox=self.sandbox, metadata_embedder=self.metadata_embedder),
            RetrieveSimilarChunks(sandbox=self.sandbox),
            ValidateCleaningResults(sandbox=self.sandbox),
            GetToolHelp(sandbox=self.sandbox, metadata_embedder=self.metadata_embedder),
            RunCode(sandbox=self.sandbox),
            SaveCleanedDataframe(sandbox=self.sandbox),
            
        ]
        return tools

from smolagents import Tool
from src.client.telemetry import TelemetryManager
from openai import OpenAI
from langfuse import observe, get_client


class GetToolHelp(Tool):
    name = "GetToolHelp"
    description = "Returns detailed help and usage examples for a tool by name."
    inputs = {
        "tool_name": {"type": "string", "description": "Name of the tool to get help on"}
    }
    output_type = "string"
    help_notes = """ 
    GetToolHelp: 
    A tool that provides detailed help information and usage examples for any other tool in the system.
    Use this when you need to understand how to use a specific tool or want more details about its functionality.

    Example usage: 

    help_text = GetToolHelp().forward(tool_name="retrieve_metadata")
    """

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox
        self.telemetry = TelemetryManager()
        self.trace = self.telemetry.start_trace("GetToolHelp")
        self.trace.finish()

    @observe(name="GetToolHelp")
    def forward(self, tool_name: str) -> str:
        langfuse = get_client()
        # Dynamically check all tool classes you registered
        for tool_cls in Tool.__subclasses__():
            if tool_cls.name == tool_name:
                print("tool_cls.name", "help_notes")
                return getattr(tool_cls, "help_notes", "No help notes available for this tool.")

        langfuse.update_current_trace(user_id="cmc1u2sny0176ad07fpb9il4b")
        return "Tool not found."

"""
class AskGPT(Tool):
    name = "AskGPT"
    description = "Allows you to ask ChatGPT for help with a specific question if you get stuck."
    inputs = {
        "question": {"type": "string", "description": "A short description of the problem you are facing."}
    }
    output_type = "string"
    help_notes =  
    AskGPT:
    Use this tool when you are stuck and need help solving a specific problem.
    Be clear and concise in your question, and include any relevant tool names or observed errors.
    

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox
        self.telemetry = TelemetryManager()

    @observe(name="AskGPT")
    def forward(self, question: str) -> str:
        from langfuse import Langfuse

        # Step 1: Collect all tool descriptions
        tool_info = []
        for tool_cls in Tool.__subclasses__():
            if hasattr(tool_cls, "name") and hasattr(tool_cls, "description"):
                tool_info.append(f"{tool_cls.name}: {tool_cls.description}")

        tool_summary = "\ n".join(tool_info)

        # Step 2: Combine with user question
        composed_prompt = f
        The agent has access to the following tools:
        {tool_summary}

        The agent is stuck and has asked the following question:
        "{question}"

        Please respond with a helpful suggestion or code snippet.
        

        # Step 3: Call GPT (placeholder logic)
        # You should replace this with your actual OpenAI/GPT API call
        print("[AskGPT Prompt]\n", composed_prompt)
        return "(This is where ChatGPT would respond.)"
"""
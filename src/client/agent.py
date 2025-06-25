from src.client.telemetry import TelemetryManager
# from smolagents import CodeAgent as BaseCodeAgent
from smolagents import ToolCallingAgent
from src.utils.prompts import TCA_MAIN_PROMPT, TCA_SYSTEM_PROMPT, CHAT_PROMPT

# class CodeAgent(BaseCodeAgent):

class CustomAgent(ToolCallingAgent):  # Renamed to avoid confusion
    def __init__(self, tools, model, *args, **kwargs):
        super().__init__(
            tools=tools,
            model=model,
            *args,
            **kwargs
        )
        self.telemetry = TelemetryManager()




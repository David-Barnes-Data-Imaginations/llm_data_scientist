from src.utils.prompts import SYSTEM_PROMPT, MAIN_PROMPT, CHAT_PROMPT
from src.config import Settings
from src.client.ui.chat import ChatInterface
from src.client.telemetry import TelemetryManager
from smolagents import CodeAgent as BaseCodeAgent
from src.utils.prompts import SYSTEM_PROMPT, MAIN_PROMPT, CHAT_PROMPT

class CodeAgent(BaseCodeAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs
        )
        self.telemetry = TelemetryManager()
        #   agent.run here?

    async def run_initial_analysis(self):
        """Start the main analysis workflow"""
        return await self.process_message(MAIN_PROMPT)

    async def chat_response(self, message: str):
        """Handle interactive chat with context"""
        # Combine CHAT_PROMPT with user message for context
        contextualized_message = f"{CHAT_PROMPT}\n\nUser Question: {message}"
        return await self.process_message(contextualized_message)
# src/client/ui/chat.py
import gradio as gr
from src.client.telemetry import log_user_feedback
from src.client.agent import CodeAgent

class ChatInterface:
    def __init__(self, agent: CodeAgent):
        self.agent = agent
        self.current_trace_id = None

    async def respond(self, prompt: str, history: list) -> list:
        """Handle chat responses with telemetry"""
        response = await self.agent.process_message(prompt)
        history.append({"role": "assistant", "content": str(response)})
        return history

    def handle_feedback(self, data: gr.LikeData):
        """Process user feedback"""
        if self.current_trace_id:
            log_user_feedback(self.current_trace_id, data.liked)

    def create_interface(self) -> gr.Blocks:
        """Create and configure the Gradio interface"""
        with gr.Blocks() as chat_interface:
            chatbot = gr.Chatbot(label="Chat", type="messages")
            prompt_box = gr.Textbox(
                placeholder="Type your message...",
                label="Your message"
            )

            prompt_box.submit(
                fn=self.respond,
                inputs=[prompt_box, chatbot],
                outputs=chatbot
            )
            chatbot.like(self.handle_feedback, None, None)

        return chat_interface
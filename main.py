import asyncio, os
from fastapi import FastAPI, Request
from gradio import ChatInterface
from sse_starlette.sse import EventSourceResponse
from e2b_code_interpreter import Sandbox
from src.client.agent import CustomAgent  # Import your custom agent
from src.client.telemetry import TelemetryManager
# from smolagents.local_python_executor import LocalPythonExecutor # Used for CodeAgent
from typing import AsyncGenerator
from src.client.mcp_client import create_mcp_client, list_tools

# Initialize FastAPI
app = FastAPI()

HF_TOKEN: os.getenv('HF_TOKEN')

# Global variables for component access
sandbox = None
agent = None
chat_interface = None

@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    global sandbox, agent, chat_interface

    # Initialize sandbox
    sandbox = Sandbox()

# Upload dataset to sandbox
    with open("./src/data/tg_database.db", "rb") as f:
        dataset_path_in_sandbox = sandbox.files.write("/data/tg_database.db", f)
    with open("./src/data/metadata/turtle_games_dataset_metadata.md", "rb") as f:
        metadata_path_in_sandbox = sandbox.files.write("/data/metadata/turtle_games_dataset_metadata.md", f)

    # Install required packages in sandbox
    sandbox.commands.run("pip install smolagents")

    # Initialize MCP components and create agent
    mcp_client, mcp_tools = await create_mcp_client()
    await list_tools(mcp_tools)

    agent = CustomAgent()
    agent.telemetry = TelemetryManager()
    # Run the agent code in the sandbox

    # Initialize chat interface
    chat_interface = ChatInterface(agent)

    # Initialize chat interface
    chat_interface = ChatInterface(agent)
    try:
        # ... initialization code ...
        pass
    except Exception as e:
        # logger.error(f"Failed to initialize application: {e}")
        raise

@app.get("/stdio")
async def stdio_endpoint(request: Request) -> EventSourceResponse:
    """Server-Sent Events endpoint"""
    async def event_generator() -> AsyncGenerator[str, None]:
        while True:
            if await request.is_disconnected():
                break
            await asyncio.sleep(1)
    return EventSourceResponse(event_generator())

@app.get("/chat")
async def launch_chat():
    """Launch the chat interface"""
    if chat_interface is None:
        return {"error": "Chat interface not initialized"}
    interface = chat_interface.create_interface()
    interface.launch()
    return {"status": "Chat interface launched"}

def run_server():
    """Run the FastAPI server"""
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if sandbox:
        await sandbox.cleanup()

if __name__ == "__main__":
    run_server()
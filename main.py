import asyncio, os
from fastapi import FastAPI, Request
from gradio import ChatInterface
from sse_starlette.sse import EventSourceResponse
from e2b_code_interpreter import Sandbox
from src.client.agent import CustomAgent, ToolFactory
from src.client.telemetry import TelemetryManager
from src.utils.metadata_embedder import MetadataEmbedder
# from src.client.mcp_client import create_all_tools, list_tools
from typing import AsyncGenerator


# Retain for potential future MCP use
# from src.client.mcp_client import create_mcp_client, list_tools

# for none MCP required runs
from src.client.mcp_client import create_local_tools, list_tools


# Initialize FastAPI
app = FastAPI()

HF_TOKEN: os.getenv('HF_TOKEN')

# Global variables for component access
sandbox = None
agent = None
chat_interface = None
metadata_embedder = None

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

    # Create tool factory and tools
    tool_factory = ToolFactory(sandbox)
    tools = tool_factory.create_all_tools()

    # When using MCP, Initialize MCP components and create agent
    # mcp_client, mcp_tools = await create_mcp_client()
    # await list_tools(mcp_tools)

    # For none MCP runs
     _, local_tools = await create_local_tools()
    await list_tools(local_tools)


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
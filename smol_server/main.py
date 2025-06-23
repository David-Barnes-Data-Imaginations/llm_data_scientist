import asyncio
from fastapi import FastAPI, Request
from sse_starlette.sse import EventSourceResponse
from e2b_code_interpreter import Sandbox
from smolagents.local_python_executor import LocalPythonExecutor
from typing import AsyncGenerator

from src.client.ui.chat import ChatInterface
from src.client.mcp_client import create_mcp_client, list_tools

# Initialize FastAPI
app = FastAPI()

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
        dataset_path_in_sandbox=sandbox.files.write("/data/tg_database.db", f)
        # Upload dataset to sandbox
    with (open(".src/data/metadata/turtle_games_dataset_metadata.md", "rb") as f):
        metadata_path_in_sandbox=sandbox.files.write("/data/metadata/turtle_games_dataset_metadata.md", f)

    # Install required packages in sandbox
    sandbox.commands.run("pip install smolagents")
    
    # Set up a custom executor
    custom_executor = LocalPythonExecutor(
        ['sqlalchemy', 'random', 'sklearn', 'statistics', 'pandas',
         'itertools', 'queue', 'math'],
        max_print_outputs_length=1024
    )
    
    # Initialize MCP components
    mcp_client, mcp_tools = await create_mcp_client()
    await list_tools(mcp_tools)
    
    # Initialize agent
    agent = await create_agent(mcp_tools)
    
    # Initialize chat interface
    chat_interface = ChatInterface(agent)
    try:
        # ... initialization code ...
        pass
    except Exception as e:
        # logger.error(f"Failed to initialize application: {e}")
        raise

@app.get("/sse")
async def sse_endpoint(request: Request) -> EventSourceResponse:
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
    uvicorn.run(app, host="127.0.0.1", port=8000)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if sandbox:
        await sandbox.cleanup()

if __name__ == "__main__":
    run_server()
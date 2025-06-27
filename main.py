import asyncio, os
from fastapi import FastAPI, Request
from gradio import ChatInterface
from sse_starlette.sse import EventSourceResponse
from e2b_code_interpreter import Sandbox
from src.client.telemetry import TelemetryManager
from src.utils.metadata_embedder import MetadataEmbedder
from src.client.agent import ToolFactory, CustomAgent
from src.utils.vllm_utils import wait_for_vllm_server, start_vllm_server_background
from typing import AsyncGenerator

# Initialize FastAPI
app = FastAPI()

HF_TOKEN = os.getenv('HF_TOKEN')

# Global variables for component access
sandbox = None
agent = None
chat_interface = None
metadata_embedder = None
vllm_process = None

@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    global sandbox, agent, chat_interface, metadata_embedder, vllm_process

    # Start vLLM server first
    print("🚀 Starting vLLM server...")
    model_path = "./models/Qwen/Qwen2.5-Coder-32B.gguf"
    vllm_process = start_vllm_server_background(model_path)
    
    # Wait for vLLM to be ready
    if not wait_for_vllm_server(max_wait=120):  # 2 minutes timeout
        raise RuntimeError("Failed to start vLLM server")

    # Initialize sandbox
    print("🔧 Initializing sandbox...")
    sandbox = Sandbox()

    # Upload dataset to sandbox
    with open("./src/data/tg_database.db", "rb") as f:
        dataset_path_in_sandbox = sandbox.files.write("/data/tg_database.db", f)
    with open("./src/data/metadata/turtle_games_dataset_metadata.md", "rb") as f:
        metadata_path_in_sandbox = sandbox.files.write("/data/metadata/turtle_games_dataset_metadata.md", f)

    # Install required packages in sandbox
    sandbox.commands.run("pip install smolagents faiss-cpu openai numpy sqlalchemy pandas")

    # Initialize metadata embedder and embed metadata file
    print("📚 Setting up metadata embeddings...")
    metadata_embedder = MetadataEmbedder(sandbox)
    result = metadata_embedder.embed_metadata_file("/data/metadata/turtle_games_dataset_metadata.md")
    print(f"Metadata embedding result: {result}")

    # Create tool factory and tools
    print("🛠️ Creating tools...")
    tool_factory = ToolFactory(sandbox, metadata_embedder)
    tools = tool_factory.create_all_tools()

    # Create agent with tools and dependencies
    print("🤖 Creating agent...")
    agent = CustomAgent(tools=tools, sandbox=sandbox, metadata_embedder=metadata_embedder)
    agent.telemetry = TelemetryManager()

    # Initialize chat interface
    chat_interface = ChatInterface(agent)

    print("✅ Application startup complete!")

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
    global sandbox, vllm_process
    
    if sandbox:
        await sandbox.cleanup()
    
    if vllm_process:
        print("🛑 Stopping vLLM server...")
        vllm_process.terminate()
        vllm_process.wait()

if __name__ == "__main__":
    run_server()
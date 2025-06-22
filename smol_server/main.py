import asyncio
from fastapi import FastAPI, Request
from sse_starlette.sse import EventSourceResponse
from src.client.mcp_client import create_mcp_client, list_tools
from src.client.llm import setup_llm
from typing import AsyncGenerator

app = FastAPI()

# Initialize components
llm = setup_llm()
mcp_client, mcp_tools = create_mcp_client()

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    # List available tools on startup
    await list_tools(mcp_tools)

@app.get("/sse")
async def sse_endpoint(request: Request) -> EventSourceResponse:
    async def event_generator() -> AsyncGenerator[str, None]:
        while True:
            # Check if client is still connected
            if await request.is_disconnected():
                break
                
            # Your event generation logic here
            # You can use mcp_client and llm here
            
            await asyncio.sleep(1)  # Prevent tight loop
            
    return EventSourceResponse(event_generator())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
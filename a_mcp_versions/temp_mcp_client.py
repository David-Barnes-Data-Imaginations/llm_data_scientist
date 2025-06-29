from smolagents import ToolCollection
from mcp import StdioServerParameters
from a_mcp_versions.config import SERVER_ENDPOINT
import os
# Server parameters for MCP
server_parameters = StdioServerParameters(
    command="uvx",  # Using uvx ensures dependencies are available
    args=["--quiet", "smol_server"],
    env={"UV_PYTHON": "3.10", **os.environ},
)

async def create_mcp_client(endpoint: str = SERVER_ENDPOINT):
    """Create and initialize the MCP client using smolagents ToolCollection."""
    
    # Use smolagents ToolCollection.from_mcp - this is the correct way
    tool_collection = ToolCollection.from_mcp(
        server_parameters, 
        trust_remote_code=False  # Set based on your security requirements
    )
    
    return tool_collection, tool_collection.tools

async def list_tools(tools):
    """List all available tools."""
    for tool in tools:
        print(f"Tool: {tool.name} - {getattr(tool, 'description', 'No description')}")
    return tools
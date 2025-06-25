from smolagents import MCPClient, CodeAgent
from mcp import StdioServerParameters
import os
from src.config import SERVER_ENDPOINT

def create_mcp_client(endpoint: str = SERVER_ENDPOINT):
    """Create and initialize the MCP client."""
    mcp_client = BasicMCPClient(endpoint)
    mcp_tools = McpToolSpec(client=mcp_client)
    return mcp_client, mcp_tools

async def list_tools(mcp_tools: McpToolSpec):
    """List all available tools."""
    tools = await mcp_tools.to_tool_list_async()
    for tool in tools:
        print(tool.metadata.name, tool.metadata.description)
    return tools

async def shutdown():
    mcp_client.disconnect()

# ******* To change or adapt *****

# the CodeAgents MCP from hugging face documentation
server_parameters = StdioServerParameters(
    command="uvx",  # Using uvx ensures dependencies are available
    args=["--quiet", "server_name"],
    env={"UV_PYTHON": "3.13", **os.environ},
)

with MCPClient(server_parameters) as tools:
    agent = CodeAgent(tools=mcp_client.get_tools(), model=model, add_base_tools=True)


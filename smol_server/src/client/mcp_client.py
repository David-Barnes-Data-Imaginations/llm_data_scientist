from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
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
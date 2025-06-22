from llama_index.tools.mcp import McpToolSpec
from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult, ToolCall
from llama_index.core.workflow import Context
from llama_index.llms.base import LLM
from src.utils.prompts import SYSTEM_PROMPT

async def create_agent(tools: McpToolSpec, llm: LLM):
    """Create a function agent with the specified tools and LLM."""
    tools_list = await tools.to_tool_list_async()
    agent = FunctionAgent(
        name="Agent",
        description="An agent that can work with Our Database software.",
        tools=tools_list,
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent

async def handle_user_message(
    message_content: str,
    agent: FunctionAgent,
    agent_context: Context,
    verbose: bool = False,
):
    """Handle user messages and return agent responses."""
    handler = agent.run(message_content, ctx=agent_context)
    async for event in handler.stream_events():
        if verbose and isinstance(event, ToolCall):
            print(f"Calling tool {event.tool_name} with kwargs {event.tool_kwargs}")
        elif verbose and isinstance(event, ToolCallResult):
            print(f"Tool {event.tool_name} returned {event.tool_output}")

    response = await handler
    return str(response)
from a_mcp_versions.config import Settings
from smolagents import ToolCallingAgent

"""Initialize the Agent with the specified model parameters"""
# All tools (custom + HF Hub) go in the same parameter
agent = ToolCallingAgent(
        tools=Settings.llm_config.tools,
        model=Settings.model_name,
        max_completion_tokens=Settings.max_completion_tokens,
        temperature=Settings.temperature,
        top_p=Settings.top_p,
        top_k=Settings.top_k,
        system_prompt=Settings.system_prompt,
        planning_interval=Settings.planning_interval,
        add_base_tools=Settings.add_base_tools,
        executor_type=Settings.custom_executor,
        #               additional_authorized_imports=Settings.additional_authorized_imports,
        verbosity_level=Settings.verbosity_level,
        stream_outputs=Settings.stream_outputs
        )



async def create_local_tools():
        """Create local tools for data science work."""

        # Import your custom tools
        from tools.database_tools import DatabaseQueryTool
        from tools.data_structure_inspection_tools import DataInspectionTool
        # Add more as you build them

        # Create instances of your custom tools
        local_tools = [
                DatabaseQueryTool(),
                DataInspectionTool(),
        ]

        # Optionally add HF Hub tools
        # hf_tools = [
        #     load_tool("huggingface-tools/text-classification"),
        # ]

        # Combine all tools
        all_tools = local_tools  # + hf_tools if you want HF tools

        return None, all_tools

async def list_tools(tools):
        """List all available tools."""
        for tool in tools:
                print(f"Tool: {tool.name} - {getattr(tool, 'description', 'No description')}")
        return tools


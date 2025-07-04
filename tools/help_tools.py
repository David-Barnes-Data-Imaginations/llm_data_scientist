from smolagents import Tool


class GetToolHelp(Tool):
    name = "GetToolHelp"
    description = "Get detailed help and usage examples for tools using semantic search on embedded help notes."
    inputs = {
        "query": {"type": "string", "description": "Tool name or description of what you want to do (e.g. 'merge dataframes' or 'QuerySales')"}
    }
    output_type = "string"
    help_notes = """ 
    GetToolHelp: 
    A tool that provides detailed help information and usage examples for tools using semantic search.
    Instead of exact tool names, you can describe what you want to do and it will find relevant tools.
    Uses embeddings to match your query with the most relevant tool help documentation.

    Example usage: 

    # Search by tool name
    help_text = GetToolHelp(query="QuerySales")
    
    # Search by functionality
    help_text = GetToolHelp(query="merge two dataframes")
    
    # Search by problem description
    help_text = GetToolHelp(query="how to handle missing values")
    """

    def __init__(self, sandbox=None, metadata_embedder=None):
        super().__init__()
        self.sandbox = sandbox
        self.metadata_embedder = metadata_embedder

    def forward(self,query):
        """Get tool help using semantic search on embedded help notes."""
        try:
            from smolagents import Tool

            if not self.metadata_embedder:
                # Fallback to simple string matching if no embedder available
                for tool_cls in Tool.__subclasses__():
                    if hasattr(tool_cls, 'name') and tool_cls.name.lower() == query.lower():
                        help_notes = getattr(tool_cls, "help_notes", "No help notes available.")
                        return f"Tool: {tool_cls.name}\nDescription: {tool_cls.description}\n\nHelp Notes:\n{help_notes}"

                return f"Tool '{query}' not found. Available tools: {[tool.name for tool in Tool.__subclasses__() if hasattr(tool, 'name')]}"

            # Use embeddings to search for relevant tool help
            search_results = self.metadata_embedder.search_chunks(
                query=f"tool help: {query}",
                top_k=3,
                similarity_threshold=0.7
            )

            if not search_results:
                # Fallback: try broader search
                search_results = self.metadata_embedder.search_chunks(
                    query=query,
                    top_k=5,
                    similarity_threshold=0.5
                )

            if search_results:
                # Format the results
                help_content = "Relevant Tool Help:\n\n"
                for i, result in enumerate(search_results, 1):
                    chunk_text = result.get('text', '')
                    similarity = result.get('similarity', 0)

                    help_content += f"Result {i} (similarity: {similarity:.3f}):\n"
                    help_content += f"{chunk_text}\n\n"
                    help_content += "-" * 50 + "\n\n"

                return help_content
            else:
                # No results found
                fallback_msg = f"No tool help found for '{query}'. Try being more specific or using exact tool names.\n\n"
                fallback_msg += "Available tools:\n"

                # List available tools
                for tool_cls in Tool.__subclasses__():
                    if hasattr(tool_cls, 'name') and hasattr(tool_cls, 'description'):
                        fallback_msg += f"- {tool_cls.name}: {tool_cls.description}\n"

                return fallback_msg

        except Exception as e:
            error_message = f"Error searching tool help: {str(e)}"
            return error_message


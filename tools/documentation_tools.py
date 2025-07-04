from smolagents import Tool


class RetrieveMetadata(Tool):
    name = "RetrieveMetadata"
    description = "Search the dataset metadata for relevant information"
    inputs = {
        "query": {"type": "string", "description": "What to search for in the metadata"},
        "k": {"type": "integer", "description": "Number of results to return (default: 3)", "nullable": True}
    }
    output_type = "string"
    help_notes = """ 
    RetrieveMetadata: 
    A tool that searches through the dataset's metadata to find relevant information based on your query.
    Use this when you need to understand the dataset structure, field meanings, or any documented information about the data.

    Example usage: 

    metadata_info = RetrieveMetadata().forward(query="customer demographics", k=5)
    """

    def __init__(self, sandbox=None, metadata_embedder=None):
        super().__init__()
        self.sandbox = sandbox
        self.metadata_embedder = metadata_embedder

    def forward(self, query, k=3):
        """
        Args:
            query (str): What to search for in the metadata
            k (int): Number of results to return

        Returns:
            str: Relevant metadata chunks
        """
        if not self.metadata_embedder:
            return "Error: Metadata embedder not available"

        results = self.metadata_embedder.search_metadata(query, k)

        if not results:
            return "No relevant metadata found"

        response = "Relevant metadata:\n\n"
        for i, result in enumerate(results, 1):
            response += f"**Result {i}** (similarity: {result['similarity_score']:.3f})\n"
            response += f"{result['content']}\n\n"

        return response

class DocumentLearningInsights(Tool):
    name = "DocumentLearningInsights"
    description = "Logs and embeds the agent's insights from a data chunk, storing both the markdown/JSON summaries and vector embeddings."
    inputs = {
        "notes": {"type": "string", "description": "The agent's reflections on the current chunk"}
    }
    output_type = "string"
    help_notes = """ 
    DocumentLearningInsights: 
    A tool that allows you to document your insights, observations, and learnings about a data chunk.
    These notes are stored both as readable markdown/JSON and as vector embeddings for future retrieval.
    Use this to record important findings that might be useful later in your analysis.

    Example usage: 

    result = DocumentLearningInsights().forward(notes="This chunk contains customer data with several outliers in the age column. Most values are between 25-45 years.")
    """

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

        # File paths
        self.agent_notes_index_path = "embeddings/agent_notes_index.faiss"
        self.agent_notes_store_path = "embeddings/agent_notes_store.json"


    def forward(self, notes):
        """
        Args:
            notes (str): The agent's reflections on the current chunk.

        Returns:
            str: Confirmation message including the assigned chunk number.
        """
        if not self.sandbox:
            return "Error: Sandbox not available"

        # Get chunk number, triple logic temp added for testing
        index_path = "insights/chunk_index.txt"
        try:
            current_index = int(self.sandbox.files.read(index_path).decode().strip())
            chunk_number = current_index + 1
            self.sandbox.files.write(index_path, str(chunk_number).encode())
        except:
            chunk_number = 0

        # Save markdown and JSON versions
        md_path = f"insights/chunk_{chunk_number}.md"
        json_path = f"insights/chunk_{chunk_number}.json"

        md_content = f"""## Analysis Insights - Chunk {chunk_number}
        

### Agent Notes
{notes}
"""
        json_content = {
            "chunk": chunk_number,
            "notes": notes,
            "type": "agent_notes"
        }

        try:
            import os
            import json
            import numpy as np
            from openai import OpenAI
            
            # Using simple numpy-based similarity instead of faiss
            
            # Initialize OpenAI client
            openai_api_key = os.getenv("OPENAI_API_KEY")
            openai_client = OpenAI(api_key=openai_api_key)
            
            # Create embedding
            response = openai_client.embeddings.create(
                input=notes,
                model="text-embedding-3-small"
            )
            embedding = response.data[0].embedding


            # Simple storage with full embeddings (no faiss needed)
            chunk_data = {
                "chunk": chunk_number,
                "notes": notes,
                "embedding": embedding,  # Store full embedding for similarity search
                "type": "agent_notes"
            }
            
            try:
                store_data = self.sandbox.files.read(self.agent_notes_store_path).decode()
                agent_store = json.loads(store_data)
            except:
                agent_store = []
            
            agent_store.append(chunk_data)

            # Save everything
            self.sandbox.files.write(md_path, md_content.encode())
            self.sandbox.files.write(json_path, json.dumps(json_content, indent=2).encode())
            self.sandbox.files.write(index_path, str(chunk_number).encode())

            # Save embeddings as simple JSON
            store_json = json.dumps(agent_store, indent=2)
            self.sandbox.files.write(self.agent_notes_store_path, store_json.encode())

            return f"Logged and embedded notes for chunk {chunk_number}"

        except Exception as e:
            return f"Error processing notes: {e}"



class RetrieveSimilarChunks(Tool):
    name = "RetrieveSimilarChunks"
    description = "Retrieves the most similar past notes based on semantic similarity."
    inputs = {
        "query": {"type": "string", "description": "The query or current goal the agent is working on"},
        "num_results": {"type": "integer", "description": "Number of top similar chunks to return", "optional": True, "nullable": True}
    }
    output_type = "object"  # Returns list of dictionaries
    help_notes = """ 
    RetrieveSimilarChunks: 
    A tool that finds previously documented insights that are semantically similar to your current query.
    Use this when you want to reference past observations or findings that might be relevant to your current task.
    This helps maintain consistency in your analysis and build upon previous work.

    Example usage: 

    similar_chunks = RetrieveSimilarChunks().forward(query="customer age distribution patterns", num_results=3)
    """

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    def forward(self, query, num_results=3):
        """
        Args:
            query (str): The query or current goal the agent is working on.
            num_results (int): Number of top similar chunks to return.

        Returns:
            list of dict: Each item contains { "chunk": int, "notes": str }
        """
        if not self.sandbox:
            return [{"chunk": 0, "notes": "Sandbox not available"}]

        import json
        
        # Simple fallback - just return stored notes without similarity search for now
        try:
            store_data = self.sandbox.files.read("embeddings/agent_notes_store.json").decode()
            agent_store = json.loads(store_data)
            
            # Return first few items limited by num_results
            results = []
            limit = min(num_results, len(agent_store))
            for i in range(limit):
                item = agent_store[i]
                results.append({
                    "chunk": item.get("chunk", i),
                    "notes": item.get("notes", "No notes available")
                })
            
            return results if results else [{"chunk": 0, "notes": "No previous notes found"}]
        except:
            return [{"chunk": 0, "notes": "No previous notes found"}]

class ValidateCleaningResults(Tool):
    name = "ValidateCleaningResults"
    description = "Validates cleaning results for a chunk and writes markdown and JSON logs."
    inputs = {
        "chunk_number": {"type": "integer", "description": "The chunk number being validated"},
        "original_chunk": {"type": "object", "description": "The original data chunk"},
        "cleaned_chunk": {"type": "object", "description": "The cleaned data chunk"}
    }
    output_type = "object"  # Returns dictionary with validation results
    help_notes = """ 
     ValidateCleaningResults: 
     A tool you can use after you have cleaned a chunk, it allows you to check if your dataframe is clean.
     It is the only way to submit your cleaned dataframe so must be used with every chunk.

     Example usage: 

     result = ValidateData().forward(chunk=my_df, name="df_validated_chunk1")
    """

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox


    def forward(self, chunk_number, original_chunk, cleaned_chunk):
        """
        Args:
            chunk_number (int): The chunk number being validated
            original_chunk (list[dict]): The original data chunk
            cleaned_chunk (list[dict]): The cleaned data chunk

        Returns:
            dict: { "logical_issues": [...], "stat_summary": {...}, "suggested_fixes": [...] }
        """
        import json
        index_path = "insights/chunk_index.txt"
        current_index = int(self.sandbox.files.read(index_path).decode().strip())
        chunk_number = current_index

        issues = []
        suggestions = []

        # Very simple rule-based example for checks
        for row in cleaned_chunk:
            if row.get("age", 0) < 0 or row.get("age", 0) > 120:
                issues.append(row)

        if issues:
            suggestions.append("Review outliers in 'age'; some extreme values remain.")

        summary = {
            "logical_issues": issues,
            "stat_summary": {
                "original_count": len(original_chunk),
                "cleaned_count": len(cleaned_chunk)
            },
            "suggested_fixes": suggestions
        }

        md = f"""## Validation Report - Chunk {chunk_number}

### Logical Issues
{json.dumps(issues, indent=2)}

### Suggestions
{json.dumps(suggestions, indent=2)}
"""

        self.sandbox.files.write(f"validation/chunk_{chunk_number}.md", md.encode())
        self.sandbox.files.write(f"validation/chunk_{chunk_number}.json", json.dumps(summary, indent=2).encode())

        return summary

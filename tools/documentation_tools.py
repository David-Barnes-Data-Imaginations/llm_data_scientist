from smolagents import Tool
import pandas as pd
import json
import faiss
import numpy as np
import os
from src.client.telemetry import TelemetryManager
from openai import OpenAI

embedding_index = faiss.IndexFlatL2(1536)  # Using OpenAI's text-embedding-3-small
metadata_store = []
metadata_store_path = "embeddings/metadata_store.json"
agent_notes_index_path = "embeddings/agent_notes_index.faiss"
agent_notes_store_path = "embeddings/agent_notes_store.json"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # for embeddings


class GetToolHelp(Tool):
    name = "get_tool_help"
    description = "Returns detailed help and usage examples for a tool by name."
    inputs = {
        "tool_name": {"type": "string", "description": "Name of the tool to get help on"}
    }
    output_type = "string"
    help_notes = """ 
    GetToolHelp: 
    A tool that provides detailed help information and usage examples for any other tool in the system.
    Use this when you need to understand how to use a specific tool or want more details about its functionality.

    Example usage: 

    help_text = GetToolHelp().forward(tool_name="retrieve_metadata")
    """

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox
        self.telemetry = TelemetryManager()
        self.trace = self.telemetry.start_trace("GetToolHelp")
        self.trace.finish()

    def forward(self, tool_name: str) -> str:
        # Dynamically check all tool classes you registered
        for tool_cls in Tool.__subclasses__():
            if tool_cls.name == tool_name:
                print("tool_cls.name", "help_notes")
                return getattr(tool_cls, "help_notes", "No help notes available for this tool.")
        return "Tool not found."


class RetrieveMetadata(Tool):
    name = "retrieve_metadata"
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

        self.telemetry = TelemetryManager()
        self.trace = self.telemetry.start_trace("retrieve_metadata")
        self.trace.add_input("query", "What to search for in the metadata")
        self.trace.add_input("k", "Number of results to return (default: 3)")
        self.trace.add_output("results", "Relevant metadata chunks")
        self.trace.finish()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = OpenAI(api_key=openai_api_key)

    def forward(self, query: str, k: int = 3) -> str:
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
    name = "document_learning_insights"
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
        openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = OpenAI(api_key=openai_api_key)

        # File paths
        self.agent_notes_index_path = "embeddings/agent_notes_index.faiss"
        self.agent_notes_store_path = "embeddings/agent_notes_store.json"

        self.telemetry = TelemetryManager()
        self.trace = self.telemetry.start_trace("document_learning_insights")
        self.trace.add_input("notes", "The agent's reflections on the current chunk")
        self.trace.add_output("confirmation", "Confirmation message including the assigned chunk number")
        self.trace.finish()

    def forward(self, notes: str) -> str:
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
            # Create embedding
            response = self.openai_client.embeddings.create(
                input=notes,
                model="text-embedding-3-small"
            )
            embedding = response.data[0].embedding

            # Load or create agent notes index
            try:
                index_data = self.sandbox.files.read(self.agent_notes_index_path)
                agent_index = faiss.deserialize_index(np.frombuffer(index_data, dtype=np.uint8))

                store_data = self.sandbox.files.read(self.agent_notes_store_path).decode()
                agent_store = json.loads(store_data)
            except:
                # Create new index
                dimension = len(embedding)
                agent_index = faiss.IndexFlatIP(dimension)
                agent_store = []

            # Add to index and store
            agent_index.add(np.array([embedding]).astype('float32'))
            agent_store.append(json_content)

            # Save everything
            self.sandbox.files.write(md_path, md_content.encode())
            self.sandbox.files.write(json_path, json.dumps(json_content, indent=2).encode())
            self.sandbox.files.write(index_path, str(chunk_number).encode())

            # Save embeddings
            index_bytes = faiss.serialize_index(agent_index).tobytes()
            self.sandbox.files.write(self.agent_notes_index_path, index_bytes)
            store_json = json.dumps(agent_store, indent=2)
            self.sandbox.files.write(self.agent_notes_store_path, store_json.encode())

            return f"Logged and embedded notes for chunk {chunk_number}"

        except Exception as e:
            return f"Error processing notes: {e}"


class RetrieveSimilarChunks(Tool):
    name = "retrieve_similar_chunks"
    description = "Retrieves the most similar past notes based on semantic similarity."
    inputs = {
        "query": {"type": "string", "description": "The query or current goal the agent is working on"},
        "top_k": {"type": "integer", "description": "Number of top similar chunks to return", "optional": True, "nullable": True}
    }
    output_type = "object"  # Returns list of dictionaries
    help_notes = """ 
    RetrieveSimilarChunks: 
    A tool that finds previously documented insights that are semantically similar to your current query.
    Use this when you want to reference past observations or findings that might be relevant to your current task.
    This helps maintain consistency in your analysis and build upon previous work.

    Example usage: 

    similar_chunks = RetrieveSimilarChunks().forward(query="customer age distribution patterns", top_k=3)
    """

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox
        openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.telemetry = TelemetryManager()
        self.trace = self.telemetry.start_trace("retrieve_similar_chunks")
        self.trace.add_input("query", "The query or current goal the agent is working on")
        self.trace.add_input("top_k", "Number of top similar chunks to return")
        self.trace.add_output("results", "List of dictionaries with chunk information")
        self.trace.finish()

    def forward(self, query: str, top_k: int = 3) -> list:
        """
        Args:
            query (str): The query or current goal the agent is working on.
            top_k (int): Number of top similar chunks to return.

        Returns:
            list of dict: Each item contains { "chunk": int, "notes": str }
        """
        if not self.sandbox:
            return 1
            "Error: Sandbox not available"

        # Embed the query
        response = self.openai_client.embeddings.create(
            input=[query],
            model="text-embedding-3-small"
        ).data[0].embedding

        query_vector = np.array([query]).astype("float32")

        # Search in FAISS
        distances, indices = embedding_index.search(query_vector, top_k)

        results = []
        for idx in indices[0]:
            if idx < len(metadata_store):
                chunk_info = metadata_store[idx]
                chunk_num = chunk_info.get("chunk", idx)
                md_path = f"insights/chunk_{chunk_num}.md"

                try:
                    notes = self.sandbox.files.read(md_path).decode()
                except:
                    notes = "(Could not read notes.)"

                results.append({
                    "chunk": chunk_num,
                    "notes": notes.strip()
                })

        return results

class ValidateCleaningResults(Tool):
    name = "validate_cleaning_results"
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

        self.telemetry = TelemetryManager()
        self.trace = self.telemetry.start_trace("validate_cleaning_results")
        self.trace.add_input("chunk_number", "The chunk number being validated")
        self.trace.add_input("original_chunk", "The original data chunk")
        self.trace.add_input("cleaned_chunk", "The cleaned data chunk")
        self.trace.add_output("validation_results", "Dictionary with validation results")
        self.trace.finish()

    def forward(self, chunk_number: int, original_chunk: list[dict], cleaned_chunk: list[dict]) -> dict:
        """
        Args:
            chunk_number (int): The chunk number being validated
            original_chunk (list[dict]): The original data chunk
            cleaned_chunk (list[dict]): The cleaned data chunk

        Returns:
            dict: { "logical_issues": [...], "stat_summary": {...}, "suggested_fixes": [...] }
        """
        if not self.sandbox:
            return "Error: Sandbox not available"

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

class SaveCleanedDataframe(Tool):
    name = "save_cleaned_dataframe"
    description = "Saves the cleaned DataFrame to a CSV in the sandbox."
    inputs = {
        "df": {"type": "object", "description": "The cleaned DataFrame"},
        "filename": {"type": "string", "description": "File name for the CSV output", "optional": True, "nullable": True}
    }
    output_type = "string"  # Returns confirmation message
    help_notes = """ 
    SaveCleanedDataframe: 
    A tool that saves your cleaned pandas DataFrame to a CSV file in the sandbox environment.
    Use this when you've completed your data cleaning and want to save the results for further analysis or export.

    Example usage: 

    result = SaveCleanedDataframe().forward(df=cleaned_dataframe, filename="customer_data_cleaned.csv")
    """

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

        self.telemetry = TelemetryManager()
        self.trace = self.telemetry.start_trace("save_cleaned_dataframe")
        self.trace.add_input("df", "The cleaned DataFrame")
        self.trace.add_input("filename", "File name for the CSV output")
        self.trace.add_output("confirmation", "Confirmation message")
        self.trace.finish()

    def forward(self, df: pd.DataFrame, filename: str = "tg_reviews_cleaned.csv") -> str:
        """
        Args:
            df (pd.DataFrame): The cleaned DataFrame
            filename (str): File name for the CSV output

        Returns:
            str: Confirmation message
        """
        csv_bytes = df.to_csv(index=False).encode()

        if self.sandbox:
            self.sandbox.files.write(filename, csv_bytes)
            return f"Saved cleaned DataFrame to sandbox file: {filename}"
        else:
            with open(filename, "wb") as f:
                f.write(csv_bytes)
            return f"Saved cleaned DataFrame locally: {filename}"

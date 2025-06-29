from ray.llm._internal.serve.config_generator.utils import gpu
from smolagents import Tool
import pandas as pd
import json
import faiss
import numpy as np
import os
from src.client.telemetry import TelemetryManager


embedding_index = faiss.IndexFlatL2(1536)  # Using OpenAI's text-embedding-3-small
metadata_store = []
metadata_store_path = "embeddings/metadata_store.json"
agent_notes_index_path = "embeddings/agent_notes_index.faiss"
agent_notes_store_path = "embeddings/agent_notes_store.json"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # for embeddings


class RetrieveMetadata(Tool):
    name = "retrieve_metadata"
    description = "Search the dataset metadata for relevant information"
    inputs = {
        "query": {"type": "string", "description": "What to search for in the metadata"},
        "k": {"type": "integer", "description": "Number of results to return (default: 3)"}
    }
    output_type = "string"

    def __init__(self, sandbox=None, metadata_embedder=None):
        super().__init__()
        self.sandbox = sandbox
        self.metadata_embedder = metadata_embedder

        self.telemetry = TelemetryManager()
        self.trace = self.telemetry.start_trace("retrieve_metadata")
        self.trace.add_input("query", "What to search for in the metadata")
        self.trace.add_input("k", "Number of results to return (default: 3)")
        self.trace.add_output("results", "Relevant metadata chunks")
        self.trace.end()

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
    description = "Logs the agent's insights from a data chunk, assigns a chunk number automatically, and stores both the markdown and JSON summaries with embeddings."
    inputs = {
        "notes": {"type": "string", "description": "The agent's reflections on the current chunk"}
    }
    output_type = "string"

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

        self.telemetry = TelemetryManager()
        self.trace = self.telemetry.start_trace("document_learning_insights")
        self.trace.add_input("notes", "The agent's reflections on the current chunk")
        self.trace.add_output("confirmation", "Confirmation message including the assigned chunk number")
        self.trace.end()

    def forward(self, notes: str) -> str:
        """
        Args:
            notes (str): The agent's reflections on the current chunk.

        Returns:
            str: Confirmation message including the assigned chunk number.
        """
        if not self.sandbox:
            return "Error: Sandbox not available"

        index_path = "insights/chunk_index.txt"

        # Read and increment chunk number
        try:
            current_index = int(self.sandbox.files.read(index_path).decode().strip())
            chunk_number = current_index + 1
        except:
            chunk_number = 0  # First chunk

        # Write .md and .json files
        md_path = f"insights/chunk_{chunk_number}.md"
        json_path = f"insights/chunk_{chunk_number}.json"

        md_content = f"""## Analysis Insights - Chunk {chunk_number}

### Agent Notes
{notes}
"""
        json_content = {
            "chunk": chunk_number,
            "notes": notes
        }

        self.sandbox.files.write(md_path, md_content.encode())
        self.sandbox.files.write(json_path, json.dumps(json_content, indent=2).encode())

        # Save updated index
        self.sandbox.files.write(index_path, str(chunk_number).encode())

        return f"Logged and embedded notes for chunk {chunk_number}."

class EmbedAndStore(Tool):
    name = "embed_and_store"
    description = "Embeds agent notes and stores them separately from metadata"
    inputs = {
        "notes": {"type": "string", "description": "The agent's notes to embed"},
        "metadata": {"type": "dict", "description": "Optional metadata about the notes (chunk number, etc.)", "optional": True, "nullable": True}
    }
    output_type = "string"

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

        self.telemetry = TelemetryManager()
        self.trace = self.telemetry.start_trace("embed_and_store")
        self.trace.add_input("notes", "The agent's notes to embed")
        self.trace.add_input("metadata", "Optional metadata about the notes (chunk number, etc.)")
        self.trace.add_output("confirmation", "Confirmation message")
        self.trace.end()

    def forward(self, notes: str, metadata: dict = None) -> str:
        """
        Args:
            notes (str): The agent's notes to embed
            metadata (dict, optional): Metadata about the notes (chunk number, etc.)

        Returns:
            str: Confirmation message
        """
        if not self.sandbox:
            return "Error: Sandbox not available"

        if metadata is None:
            metadata = {"type": "agent_notes", "chunk": 0}

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

            # Add to index
            agent_index.add(np.array([embedding]).astype('float32'))

            # Add to store
            metadata.update({
                "content": notes,
                "type": "agent_notes"
            })
            agent_store.append(metadata)

            # Save back to sandbox
            index_bytes = faiss.serialize_index(agent_index).tobytes()
            self.sandbox.files.write(self.agent_notes_index_path, index_bytes)

            store_json = json.dumps(agent_store, indent=2)
            self.sandbox.files.write(self.agent_notes_store_path, store_json.encode())

            return f"Embedded and stored agent notes (chunk {metadata.get('chunk', 'unknown')})"

        except Exception as e:
            return f"Error embedding notes: {e}"

class RetrieveSimilarChunks(Tool):
    name = "retrieve_similar_chunks"
    description = "Retrieves the most similar past notes based on semantic similarity."
    inputs = {
        "query": {"type": "string", "description": "The query or current goal the agent is working on"},
        "top_k": {"type": "integer", "description": "Number of top similar chunks to return", "optional": True, "nullable": True}
    }
    output_type = "list"  # Returns list of dictionaries

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

        self.telemetry = TelemetryManager()
        self.trace = self.telemetry.start_trace("retrieve_similar_chunks")
        self.trace.add_input("query", "The query or current goal the agent is working on")
        self.trace.add_input("top_k", "Number of top similar chunks to return")
        self.trace.add_output("results", "List of dictionaries with chunk information")
        self.trace.end()

    def forward(self, query: str, top_k: int = 3) -> list:
        """
        Args:
            query (str): The query or current goal the agent is working on.
            top_k (int): Number of top similar chunks to return.

        Returns:
            list of dict: Each item contains { "chunk": int, "notes": str }
        """
        if not self.sandbox:
            return "Error: Sandbox not available"

        # Embed the query
        query_embed = openai_client.embeddings.create(
            input=[query],
            model="text-embedding-3-small"
        ).data[0].embedding

        query_vector = np.array([query_embed]).astype("float32")

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
        "original_chunk": {"type": "list", "description": "The original data chunk"},
        "cleaned_chunk": {"type": "list", "description": "The cleaned data chunk"}
    }
    output_type = "dict"  # Returns dictionary with validation results

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

        self.telemetry = TelemetryManager()
        self.trace = self.telemetry.start_trace("validate_cleaning_results")
        self.trace.add_input("chunk_number", "The chunk number being validated")
        self.trace.add_input("original_chunk", "The original data chunk")
        self.trace.add_input("cleaned_chunk", "The cleaned data chunk")
        self.trace.add_output("validation_results", "Dictionary with validation results")
        self.trace.end()

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

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

        self.telemetry = TelemetryManager()
        self.trace = self.telemetry.start_trace("save_cleaned_dataframe")
        self.trace.add_input("df", "The cleaned DataFrame")
        self.trace.add_input("filename", "File name for the CSV output")
        self.trace.add_output("confirmation", "Confirmation message")
        self.trace.end()

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

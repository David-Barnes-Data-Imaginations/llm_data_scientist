from ray.llm._internal.serve.config_generator.utils import gpu
from smolagents import Tool
import pandas as pd
import json
from e2b import Sandbox # Can replace with docker sandbox class
from openai import OpenAI
import faiss
import numpy as np
import os

from main import sandbox

embedding_index = faiss.IndexFlatL2(1536)  # Using OpenAI's text-embedding-3-small
metadata_store = []
metadata_store_path = "embeddings/metadata_store.json"
openai_client = OpenAI()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # for embeddings

"""
 The 'MultiStepAgent' tool class for the functions used by the agent. Subclass this and implement the `forward` method as well as the
following class attributes:

- **description** (`str`) -- A short description of what your tool does, the inputs it expects and the output(s) it
  will return. For instance 'This is a tool that downloads a file from a `url`. It takes the `url` as input, and
  returns the text contained in the file'.
- **name** (`str`) -- A performative name that will be used for your tool in the prompt to the agent. For instance
  `"text-classifier"` or `"image_generator"`.
- **inputs** (`Dict[str, Dict[str, Union[str, type, bool]]]`) -- The dict of modalities expected for the inputs.
  It has one `type`key and a `description`key.
  This is used by `launch_gradio_demo` or to make a nice space from your tool, and also can be used in the generated
  description for your tool.
- **output_type** (`type`) -- The type of the tool output. This is used by `launch_gradio_demo`
  or to make a nice space from your tool, and also can be used in the generated description for your tool.

You can also override the method [`~Tool.setup`] if your tool has an expensive operation to perform before being
usable (such as loading a model). [`~Tool.setup`] will be called the first time you use your tool, but not at
instantiation.

{%- for tool in tools.values() %}
- {{ tool.name }}: {{ tool.description }}
Takes inputs: {{tool.inputs}}
Returns an output of type: {{tool.output_type}}
{%- endfor %}
"""

class RetrieveMetadata(Tool):
    name = "retrieve_metadata"
    description = "Search the dataset metadata for relevant information"

    def __init__(self, sandbox=None, metadata_embedder=None):
        super().__init__()
        self.sandbox = sandbox
        self.metadata_embedder = metadata_embedder

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
    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

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

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        # Separate paths for agent notes
        self.agent_notes_index_path = "embeddings/agent_notes_index.faiss"
        self.agent_notes_store_path = "embeddings/agent_notes_store.json"

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
    name = "RetrieveSimilarChunks"
    description = "Retrieves the most similar past notes based on semantic similarity."
    def __init__(self, sandbox=None, query: str = None, top_k: int = 3, metadata: dict = None):
        super().__init__()
        self.sandbox = sandbox

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
    name = "ValidateCleaningResults"
    description = "Validates cleaning results for a chunk and writes markdown and JSON logs."

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

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

class save_cleaned_dataframe(Tool):
    name = "save_cleaned_dataframe"
    description = "Saves the cleaned DataFrame to a CSV in the sandbox."

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

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

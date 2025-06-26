from ray.llm._internal.serve.config_generator.utils import gpu
from smolagents import tool
import pandas as pd
import json
from e2b import Sandbox # Can replace with docker sandbox class
from openai import OpenAI
import faiss
import numpy as np
import os
embedding_index = faiss.IndexFlatL2(1536)  # Using OpenAI's text-embedding-3-small
metadata_store = []
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

# @tool
class document_learning_insights(str, Sandbox)
    name="document_learning_insights"
    description = "Logs the agent's insights from a data chunk, assigns a chunk number automatically, and stores both the markdown and JSON summaries with embeddings."
    inputs: {str, Sandbox}
    output_type: str = str
    """
    
        notes (str): The agent's reflections on the current chunk.
        sandbox (Sandbox): E2B sandbox instance.

    Returns:
        str: Confirmation message including the assigned chunk number.
    """
    index_path = "insights/chunk_index.txt"

    # Read and increment chunk number
    try:
        current_index = int(sandbox.files.read(index_path).decode().strip())
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

    sandbox.files.write(md_path, md_content.encode())
    sandbox.files.write(json_path, json.dumps(json_content, indent=2).encode())

    # Store embedding
    embed_and_store(notes, metadata={"chunk": chunk_number})

    # Save updated index
    sandbox.files.write(index_path, str(chunk_number).encode())

    return f"Logged and embedded notes for chunk {chunk_number}."

# @tool
class embed_and_store(text: str, metadata: dict = None, sandbox: Sandbox):
    embedding = openai_client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    ).data[0].embedding

    embedding_index.add(np.array([embedding]).astype("float32"))
    metadata_store.append(metadata or {})

    sandbox.files.write(
        f"embeddings/chunk_{metadata['chunk']}.json",
        json.dumps({
            "embedding": embedding,
            "metadata": metadata
        }, indent=2).encode()
    )


@tool
def retrieve_similar_chunks(query: str, top_k: int = 3, sandbox: Sandbox) -> list:
    """
    Retrieves the most similar past notes based on semantic similarity.

    Parameters:
        sandbox:
        query (str): The query or current goal the agent is working on.
        top_k (int): Number of top similar chunks to return.

    Returns:
        list of dict: Each item contains { "chunk": int, "notes": str }
    """
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
                notes = sandbox.files.read(md_path).decode()
            except:
                notes = "(Could not read notes.)"

            results.append({
                "chunk": chunk_num,
                "notes": notes.strip()
            })

    return results

@tool
def validate_cleaning_results(chunk_number: int, original_chunk: list[dict], cleaned_chunk: list[dict], sandbox: Sandbox) -> dict:
    """
    Validates cleaning results for a chunk and writes markdown and JSON logs.

    Returns:
        dict: { "logical_issues": [...], "stat_summary": {...}, "suggested_fixes": [...] }#

    # Very simple rule-based example for checks
    for row in cleaned_chunk:
        if row.get("age", 0) < 0 or row.get("age", 0) > 120:
            issues.append(row)

    if issues:
        suggestions.append("Review outliers in 'age'; some extreme values remain.")

    """
    issues = []
    suggestions = []

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

    sandbox.files.write(f"validation/chunk_{chunk_number}.md", md.encode())
    sandbox.files.write(f"validation/chunk_{chunk_number}.json", json.dumps(summary, indent=2).encode())

    return summary

@tool
def save_cleaned_dataframe(df: pd.DataFrame, filename: str = "tg_reviews_cleaned.csv", sandbox: Sandbox = None) -> str:
    """
    Saves the cleaned DataFrame to a CSV in the sandbox.

    Parameters:
        df (pd.DataFrame): The cleaned DataFrame
        filename (str): File name for the CSV output
        sandbox (Sandbox, optional): If present, will write into sandbox FS

    Returns:
        str: Confirmation message
    """
    csv_bytes = df.to_csv(index=False).encode()

    if sandbox:
        sandbox.files.write(filename, csv_bytes)
        return f"Saved cleaned DataFrame to sandbox file: {filename}"
    else:
        with open(filename, "wb") as f:
            f.write(csv_bytes)
        return f"Saved cleaned DataFrame locally: {filename}"

import json
from smolagents import tool
from e2b_code_interpreter import Sandbox
import pandas as pd

import json
from e2b import Sandbox  # Or your own sandbox class
from your_embedding_module import embed_and_store  # Update path if needed

@tool
def document_learning_insights(notes: str, sandbox: Sandbox) -> str:
    """
    Logs the agent's insights from a data chunk, assigns a chunk number automatically,
    and stores both the markdown and JSON summaries with embeddings.

    Parameters:
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

from openai import OpenAI  # Or your wrapper
import faiss

embedding_index = faiss.IndexFlatL2(1536)  # If using OpenAI's text-embedding-3-small
metadata_store = []

def embed_and_store(text: str, metadata: dict = None):
    embedding = openai_client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    ).data[0].embedding

    embedding_index.add(np.array([embedding]).astype("float32"))
    metadata_store.append(metadata or {})


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

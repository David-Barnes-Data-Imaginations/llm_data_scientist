import asyncio, os, base64
from fastapi import FastAPI, Request
from e2b_code_interpreter import Sandbox
from src.client.telemetry import TelemetryManager
from src.utils.metadata_embedder import MetadataEmbedder
from src.client.agent import ToolFactory, CustomAgent
from src.client.ui.chat import GradioUI
from src.utils.ollama_utils import wait_for_ollama_server, start_ollama_server_background, pull_model
from openai import OpenAI

HF_TOKEN = os.getenv('HF_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LANGFUSE_PUBLIC_KEY= os.getenv('LANGFUSE_PUBLIC_KEY')
LANGFUSE_SECRET_KEY= os.getenv('LANGFUSE_SECRET_KEY')
LANGFUSE_AUTH=base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()
import pandas as pd
import numpy as np
from typing import Dict

os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://cloud.langfuse.com/api/public/otel" # EU data region
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

# Global memory (replace these with a controlled registry in production / CA via the below 'with open' etc..)
dataframe_store: Dict[str, pd.DataFrame] = {}
global sandbox, agent, chat_interface, metadata_embedder

# In your main function:
def main():

    sandbox = Sandbox()

    # Upload dataset to sandbox
    with open("./src/data/tg_database.db", "rb") as f:
        dataset_path_in_sandbox = sandbox.files.write("/data/tg_database.db", f)
    with open("./src/data/metadata/turtle_games_dataset_metadata.md", "r") as f:
        metadata_path_in_sandbox = sandbox.files.write("/data/metadata/turtle_games_dataset_metadata.md", f)

    # Install required packages in sandbox
    sandbox.commands.run("pip install smolagents faiss-cpu openai numpy sqlalchemy pandas imbalanced-learn")

    # Initialize metadata embedder and embed metadata file
    print("📚 Setting up metadata embeddings...")
    metadata_embedder = MetadataEmbedder(sandbox)
    result = metadata_embedder.embed_metadata_file("/data/metadata/turtle_games_dataset_metadata.md")
    print(f"Metadata embedding result: {result}")

    # Create agent, tool factory and tools
    print("🛠️ Creating tools...")
    tool_factory = ToolFactory(sandbox, metadata_embedder)
    tools = tool_factory.create_all_tools()

    agent = CustomAgent(
        tools=tools,
        sandbox=sandbox,
        metadata_embedder=metadata_embedder,
        model_id="ollama://DeepSeek-R1-Distill"  # Specify your Ollama model
    )
    agent.telemetry = TelemetryManager()

    # Initialize chat interface using your custom GradioUI
    print("🌐 Initializing Gradio interface...")
    gradio_ui = GradioUI(agent)

    print("✅ Application startup complete!")

    # Launch the interface
    gradio_ui.launch(share=False, server_port=7860)

if __name__ == "__main__":
    main()

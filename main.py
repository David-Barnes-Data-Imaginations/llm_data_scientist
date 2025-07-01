import asyncio, os, base64
from fastapi import FastAPI, Request
from e2b_code_interpreter import Sandbox
from src.client.telemetry import TelemetryManager
from src.utils.metadata_embedder import MetadataEmbedder
from src.client.agent import ToolFactory, CustomAgent
from src.client.ui.chat import GradioUI as gradio_ui
from src.utils.ollama_utils import wait_for_ollama_server, start_ollama_server_background, pull_model
from dotenv import load_dotenv

HF_TOKEN = os.getenv('HF_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LANGFUSE_PUBLIC_KEY= os.getenv('LANGFUSE_PUBLIC_KEY')
LANGFUSE_SECRET_KEY= os.getenv('LANGFUSE_SECRET_KEY')
LANGFUSE_AUTH=base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()

# Get API key from host env
openai_api_key = os.getenv("OPENAI_API_KEY")

os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://cloud.langfuse.com/api/public/otel" # EU data region
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

# Global memory (replace these with a controlled registry in production / CA via the below 'with open' etc..)
global sandbox, agent, chat_interface, metadata_embedder

# In your main function:
def main():

    sandbox = Sandbox()

    # Upload dataset to sandbox
    with open("./src/data/tg_database.db", "rb") as f:
        dataset_path_in_sandbox = sandbox.files.write("/data/tg_database.db", f)
    with open("./src/data/metadata/turtle_games_dataset_metadata.md", "r") as f:
        metadata_path_in_sandbox = sandbox.files.write("/data/metadata/turtle_games_dataset_metadata.md", f)
    with open("./states/agent_step_log.jsonl", "rb") as f:
        sandbox.files.write("/states/agent_step_log.jsonl", f)

    # Pass all required API keys into the sandbox properly
    # (important: `export` here is shell-scoped and not enough on its own)
    sandbox.commands.run(f"echo 'OPENAI_API_KEY={openai_api_key}' >> ~/.bashrc")
    sandbox.commands.run(f"export OPENAI_API_KEY={openai_api_key}")
    
    # Add Langfuse API keys to sandbox
    sandbox.commands.run(f"echo 'LANGFUSE_PUBLIC_KEY={LANGFUSE_PUBLIC_KEY}' >> ~/.bashrc")
    sandbox.commands.run(f"export LANGFUSE_PUBLIC_KEY={LANGFUSE_PUBLIC_KEY}")
    sandbox.commands.run(f"echo 'LANGFUSE_SECRET_KEY={LANGFUSE_SECRET_KEY}' >> ~/.bashrc")
    sandbox.commands.run(f"export LANGFUSE_SECRET_KEY={LANGFUSE_SECRET_KEY}")
    
    # Also set OTEL environment variables in sandbox
    sandbox.commands.run(f"echo 'OTEL_EXPORTER_OTLP_ENDPOINT=https://cloud.langfuse.com/api/public/otel' >> ~/.bashrc")
    sandbox.commands.run(f"echo 'OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic {LANGFUSE_AUTH}' >> ~/.bashrc")
    sandbox.commands.run(f"export OTEL_EXPORTER_OTLP_ENDPOINT=https://cloud.langfuse.com/api/public/otel")
    sandbox.commands.run(f"export OTEL_EXPORTER_OTLP_HEADERS='Authorization=Basic {LANGFUSE_AUTH}'")
    
    # Verify environment variables are set
    sandbox.commands.run("echo OPENAI_API_KEY is set as: $OPENAI_API_KEY")
    sandbox.commands.run("echo LANGFUSE_PUBLIC_KEY is set as: $LANGFUSE_PUBLIC_KEY")
    sandbox.commands.run("echo LANGFUSE_SECRET_KEY is set as: $LANGFUSE_SECRET_KEY")
    
    # Install required packages in sandbox (add langfuse)
    sandbox.commands.run("pip install smolagents faiss-cpu openai numpy sqlalchemy pandas imbalanced-learn langfuse opentelemetry-api opentelemetry-sdk")

    # Initialize metadata embedder and embed metadata file
    print("📚 Setting up metadata embeddings...")
    metadata_embedder = MetadataEmbedder(sandbox)
    result = metadata_embedder.embed_metadata_file("/data/metadata/turtle_games_dataset_metadata.md")
    print(f"Metadata embedding result: {result}")

    # Create agent, tool factory and tools
    print("🛠️ Creating tools...")
    tool_factory = ToolFactory(sandbox, metadata_embedder)
    tools = tool_factory.create_all_tools()

    # Embed tool help notes
    print("📖 Embedding tool help notes...")
    help_result = metadata_embedder.embed_tool_help_notes(tools)
    print(f"Tool help embedding result: {help_result}")

    agent = CustomAgent(
        tools=tools,
        sandbox=sandbox,
        metadata_embedder=metadata_embedder,
        model_id="ollama://DeepSeek-R1-Distill"
    )
    agent.telemetry = TelemetryManager()

    # Initialize chat interface using your custom GradioUI
    print("🌐 Initializing Gradio interface...")
    ui = gradio_ui(agent)  # Pass the CustomAgent instance here!

    print("✅ Application startup complete!")

    # Launch the interface
    ui.launch(share=False, server_port=7860)


if __name__ == "__main__":
    main()
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

# Temporarily disable telemetry to focus on tool parsing issues
# os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://cloud.langfuse.com" # EU data region
# os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

# Global memory (replace these with a controlled registry in production / CA via the below 'with open' etc..)
global sandbox, agent, chat_interface, metadata_embedder
import e2b_code_interpreter

# In your main function:
def main():

    # Create E2B sandbox directly (following newer E2B examples)
    sandbox = Sandbox()

    # Upload requirements.txt and install dependencies here to give time for upload before calling install
    with open("requirements.txt", "rb") as f:
        sandbox.files.write("requirements.txt", f)
    # Upload dataset to sandbox
    with open("./src/data/tg_database.db", "rb") as f:
        dataset_path_in_sandbox = sandbox.files.write("/data/tg_database.db", f)

    with open("./src/data/metadata/turtle_games_dataset_metadata.md", "r") as f:
        metadata_path_in_sandbox = sandbox.files.write("/data/metadata/turtle_games_dataset_metadata.md", f)
    with open("./src/states/agent_step_log.jsonl", "rb") as f:
        sandbox.files.write("/states/agent_step_log.jsonl", f)

    with open("./src/data/turtle_reviews.csv", "rb") as f:
        turtle_reviews_path_in_sandbox = sandbox.files.write("/src/data/turtle_reviews.csv", f)
    with open("./src/data/turtle_sales.csv", "rb") as f:
        turtle_sales_path_in_sandbox = sandbox.files.write("/src/data/turtle_sales.csv", f)


    sandbox.commands.run("pip install -r /requirements.txt", timeout=0)

    # Pass all required API keys into the sandbox properly
    # (important: `export` here is shell-scoped and not enough on its own)
    sandbox.commands.run(f"echo 'OPENAI_API_KEY={openai_api_key}' >> ~/.bashrc")
    sandbox.commands.run(f"export OPENAI_API_KEY={openai_api_key}")

    sandbox.commands.run(f"echo 'HF_TOKEN={HF_TOKEN}' >> ~/.bashrc")
    sandbox.commands.run(f"export HF_TOKEN={HF_TOKEN}")

    # These will be re-implemented at a later date
  #  sandbox.commands.run(f"echo 'OTEL_EXPORTER_OTLP_ENDPOINT=https://cloud.langfuse.com' >> ~/.bashrc")
  #  sandbox.commands.run(f"echo 'OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic {LANGFUSE_AUTH}' >> ~/.bashrc")
  #  sandbox.commands.run(f"export OTEL_EXPORTER_OTLP_ENDPOINT=https://cloud.langfuse.com")
  #  sandbox.commands.run(f"export OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic {LANGFUSE_AUTH}")

    # Verify environment variables are set
    sandbox.commands.run("echo OPENAI_API_KEY is set as: $OPENAI_API_KEY")
    sandbox.commands.run("echo LANGFUSE_PUBLIC_KEY is set as: $LANGFUSE_PUBLIC_KEY")
    sandbox.commands.run("echo LANGFUSE_SECRET_KEY is set as: $LANGFUSE_SECRET_KEY")

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

    # Start Ollama server and pull model
    ollama_process = start_ollama_server_background()
    if not wait_for_ollama_server():
        print("❌ Failed to start Ollama server. Exiting.")
        if ollama_process:
            ollama_process.terminate()
        return
    pull_model("DeepSeek-R1")

    # Create agent with context manager support for cleanup
    agent = CustomAgent(
        tools=tools,
        sandbox=sandbox,
        metadata_embedder=metadata_embedder,
        model_id="DeepSeek-R1"
    )
    agent.telemetry = TelemetryManager()

    # Initialize chat interface using your custom GradioUI
    print("🌐 Initializing Gradio interface...")
    ui = gradio_ui(agent)  # Pass the CustomAgent instance here!

    print("✅ Application startup complete!")

    try:
        # Launch the interface
        ui.launch(share=False, server_port=7860)
    except KeyboardInterrupt:
        print("\n🛑 Received shutdown signal...")
    finally:
        # Cleanup agent resources
        print("🧹 Cleaning up agent resources...")
        agent.cleanup()
        if ollama_process:
            ollama_process.terminate()
        print("👋 Goodbye!")


if __name__ == "__main__":
    main()
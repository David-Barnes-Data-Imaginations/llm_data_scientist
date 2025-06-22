from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

# Server configuration
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
SERVER_ENDPOINT = f"http://{SERVER_HOST}:{SERVER_PORT}/sse"

# Model configuration
DEFAULT_MODEL = "deepseek-ai/deepseek-coder-1.3b-base"
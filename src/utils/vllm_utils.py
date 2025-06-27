"""
vLLM server utilities for managing the local Qwen model server.
"""
import requests
import time
import subprocess
import os
from pathlib import Path
from typing import Optional

def get_model_path() -> str:
    """Get absolute path to the model file"""
    # Get project root (assuming this file is in src/utils/)
    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / "models" / "Qwen" / "Qwen2.5-Coder-32B.gguf"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return str(model_path.absolute())

def check_vllm_server(host: str = "localhost", port: int = 8050, timeout: int = 5) -> bool:
    """Check if vLLM server is running"""
    try:
        response = requests.get(f"http://{host}:{port}/v1/models", timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False

def wait_for_vllm_server(host: str = "localhost", port: int = 8050, max_wait: int = 60) -> bool:
    """Wait for vLLM server to be ready"""
    print(f"⏳ Waiting for vLLM server at {host}:{port}...")
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if check_vllm_server(host, port):
            print("✅ vLLM server is ready!")
            return True
        
        print("⏳ Still waiting for vLLM server...")
        time.sleep(5)
    
    print("❌ vLLM server failed to start within timeout")
    return False

def start_vllm_server_background(host: str = "0.0.0.0", port: int = 8050) -> Optional[subprocess.Popen]:
    """Start vLLM server in background with absolute model path"""
    
    if check_vllm_server("localhost", port):
        print("✅ vLLM server already running")
        return None
    
    # Get absolute model path
    model_path = get_model_path()
    print(f"📂 Using model: {model_path}")
    
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,  # Use absolute path
        "--host", host,
        "--port", str(port),
        "--served-model-name", "Qwen2.5-Coder-32B",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.7",
        "--trust-remote-code",
        "--dtype", "auto",
        "--quantization", "gguf",
        "--disable-log-requests",  # Reduce noise in logs
    ]
    
    print(f"🚀 Starting vLLM server...")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return process
    except Exception as e:
        print(f"❌ Failed to start vLLM server: {e}")
        return None
#!/usr/bin/env python3

import subprocess
import sys
import os
from openai import OpenAI

def start_vllm_server():
    model_path = "./models/GemmaCoder3-12B"
    # Modify OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # vLLM server command
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", "0.0.0.0",
        "--port", "8050",
        "--served-model-name", "DeepSeek-R1-0528-Qwen3-8B",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.7",
        "--trust-remote-code",
        "--disable-log-requests",
        "--dtype", "auto",
    ]

    # shellcheck disable=SC1065
    print("Starting vLLM server...")
    print(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting vLLM server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_vllm_server()
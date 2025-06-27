#!/usr/bin/env python3

import subprocess
import sys
import os

def start_vllm_server:
    model_path = "./models/Qwen/Qwen2.5-Coder-32B.gguf"

    # vLLM server command
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--served-model-name", "Qwen2.5-Coder-32B",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.9",
        "--trust-remote-code",
        "--dtype", "auto",
        # Add quantization if needed for your GGUF
        "--quantization", "gguf",
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
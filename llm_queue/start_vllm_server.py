#!/usr/bin/env python3
"""
Helper script to start a vLLM server for testing the queue system.

This script provides an easy way to start a vLLM server with common configurations.
"""

import argparse
import subprocess
import sys
import time
import requests
from pathlib import Path


def check_vllm_installed():
    """Check if vLLM is installed."""
    try:
        import vllm
        return True
    except ImportError:
        return False


def wait_for_server(url: str, timeout: int = 60):
    """Wait for vLLM server to be ready."""
    print(f"Waiting for server at {url} to be ready...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/v1/models", timeout=5)
            if response.status_code == 200:
                models = response.json()
                print(f"✓ Server is ready! Available models: {[m['id'] for m in models.get('data', [])]}")
                return True
        except requests.exceptions.RequestException:
            pass
        
        print(".", end="", flush=True)
        time.sleep(2)
    
    print(f"\n✗ Server did not become ready within {timeout} seconds")
    return False


def main():
    parser = argparse.ArgumentParser(description="Start vLLM server for LLM queue testing")
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/DialoGPT-medium",
        help="Model to serve (default: microsoft/DialoGPT-medium)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind to (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization (default: 0.8)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model length (default: auto)"
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for server to be ready before exiting"
    )
    
    args = parser.parse_args()
    
    # Check if vLLM is installed
    if not check_vllm_installed():
        print("ERROR: vLLM is not installed.")
        print("Install it with: pip install vllm")
        print("Or for CUDA: pip install vllm[cuda]")
        sys.exit(1)
    
    # Build command
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
    ]
    
    if args.max_model_len:
        cmd.extend(["--max-model-len", str(args.max_model_len)])
    
    print("Starting vLLM Server")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Server URL: http://{args.host}:{args.port}")
    print(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    if args.max_model_len:
        print(f"Max Model Length: {args.max_model_len}")
    print("=" * 50)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 50)
    
    try:
        if args.wait:
            # Start server in background and wait for it to be ready
            process = subprocess.Popen(cmd)
            
            server_url = f"http://{args.host}:{args.port}"
            if wait_for_server(server_url):
                print(f"\n✓ vLLM server is running and ready!")
                print(f"  Server URL: {server_url}")
                print(f"  API endpoint: {server_url}/v1/chat/completions")
                print(f"  Models endpoint: {server_url}/v1/models")
                print("\nYou can now run the queue poller:")
                print(f"  python examples/llm_queue_poller.py --vllm-url {server_url}")
                print("\nPress Ctrl+C to stop the server.")
                
                # Wait for user to stop
                process.wait()
            else:
                print("Failed to start server properly.")
                process.terminate()
                sys.exit(1)
        else:
            # Just start the server (blocking)
            subprocess.run(cmd)
            
    except KeyboardInterrupt:
        print("\nShutting down vLLM server...")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

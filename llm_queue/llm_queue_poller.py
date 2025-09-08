#!/usr/bin/env python3
"""
Example script demonstrating how to use the LLM queue poller.

This script shows how to:
1. Set up and run the queue poller
2. Process requests from the queue directory
3. Handle different vLLM configurations
"""

import argparse
import asyncio
from pathlib import Path
from autocomp.common.llm_queue import LLMQueuePoller


def main():
    parser = argparse.ArgumentParser(description="LLM Queue Poller Example")
    parser.add_argument(
        "--queue-dir", 
        type=str, 
        default="./llm_queue",
        help="Directory to poll for requests (default: ./llm_queue)"
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://localhost:8000",
        help="vLLM server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of concurrent workers (default: 4)"
    )
    parser.add_argument(
        "--enable-batching",
        action="store_true",
        default=True,
        help="Enable request batching for better throughput (default: True)"
    )
    parser.add_argument(
        "--disable-batching",
        action="store_true",
        help="Disable request batching"
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Maximum number of requests per batch (default: 32)"
    )
    parser.add_argument(
        "--batch-timeout",
        type=float,
        default=2.0,
        help="Maximum time to wait for batch to fill up in seconds (default: 2.0)"
    )
    
    args = parser.parse_args()
    
    queue_dir = Path(args.queue_dir)
    
    # Handle batching arguments
    enable_batching = args.enable_batching and not args.disable_batching
    
    print("LLM Queue Poller Example")
    print("=" * 50)
    print(f"Queue directory: {queue_dir.absolute()}")
    print(f"vLLM server URL: {args.vllm_url}")
    print(f"Poll interval: {args.poll_interval}s")
    print(f"Max workers: {args.max_workers}")
    print(f"Batching enabled: {enable_batching}")
    if enable_batching:
        print(f"Max batch size: {args.max_batch_size}")
        print(f"Batch timeout: {args.batch_timeout}s")
    print("=" * 50)
    print("\nStarting poller... Press Ctrl+C to stop.")
    
    # Create and run the poller
    poller = LLMQueuePoller(
        queue_dir=queue_dir,
        vllm_url=args.vllm_url,
        poll_interval=args.poll_interval,
        max_workers=args.max_workers,
        enable_batching=enable_batching,
        max_batch_size=args.max_batch_size,
        batch_timeout=args.batch_timeout
    )
    
    try:
        poller.run()
    except KeyboardInterrupt:
        print("\nStopping poller...")


if __name__ == "__main__":
    main()

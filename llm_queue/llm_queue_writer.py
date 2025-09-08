#!/usr/bin/env python3
"""
Example script demonstrating how to use the LLM request writer.

This script shows how to:
1. Create and submit LLM requests to a queue directory
2. Check request status
3. Retrieve responses
"""

import time
import asyncio
from pathlib import Path
from autocomp.common.llm_queue import LLMRequestWriter, submit_simple_request


def main():
    # Set up queue directory
    queue_dir = Path("./llm_queue")
    writer = LLMRequestWriter(queue_dir)
    
    print(f"LLM Queue Writer Example")
    print(f"Queue directory: {queue_dir.absolute()}")
    print("-" * 50)
    
    # Example 1: Submit a simple request
    print("Example 1: Simple request")
    messages = [
        {"role": "user", "content": "Write a short poem about artificial intelligence."}
    ]
    
    request_id = writer.submit_request(
        messages=messages,
        model="llama3.1-8b",  # Replace with your model name
        temperature=0.7,
        max_tokens=200
    )
    
    print(f"Submitted request: {request_id}")
    
    # Example 2: Submit multiple requests
    print("\nExample 2: Multiple requests")
    prompts = [
        "Explain quantum computing in simple terms.",
        "Write a Python function to calculate fibonacci numbers.",
        "What are the benefits of renewable energy?",
        "Describe the process of photosynthesis."
    ]
    
    request_ids = []
    for i, prompt in enumerate(prompts):
        request_id = submit_simple_request(
            queue_dir=queue_dir,
            prompt=prompt,
            model="llama3.1-8b",
            temperature=0.5
        )
        request_ids.append(request_id)
        print(f"Request {i+1}: {request_id}")
    
    # Example 3: Complex conversation request
    print("\nExample 3: Multi-turn conversation")
    conversation = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "I need help with Python decorators."},
        {"role": "assistant", "content": "I'd be happy to help with Python decorators! What specifically would you like to know?"},
        {"role": "user", "content": "Can you show me how to create a timing decorator?"}
    ]
    
    conversation_id = writer.submit_request(
        messages=conversation,
        model="llama3.1-8b",
        temperature=0.3,
        max_tokens=500
    )
    
    print(f"Conversation request: {conversation_id}")
    
    # Monitor requests
    all_requests = [request_id] + request_ids + [conversation_id]
    print(f"\nMonitoring {len(all_requests)} requests...")
    print("Tip: Start the poller with: python examples/llm_queue_poller.py")
    
    # Check status periodically
    for _ in range(10):  # Check for up to 10 iterations
        time.sleep(2)
        
        completed = 0
        for req_id in all_requests:
            status = writer.get_request_status(req_id)
            if status == "completed":
                completed += 1
                response = writer.get_response(req_id)
                if response:
                    print(f"\n✓ Request {req_id[:8]}... completed:")
                    print(f"  Processing time: {response.processing_time:.2f}s")
                    print(f"  Response preview: {response.response[:100]}...")
        
        print(f"Status check: {completed}/{len(all_requests)} completed")
        
        if completed == len(all_requests):
            print("All requests completed!")
            break
    
    print("\nExample completed. Check the queue directory for detailed results.")


if __name__ == "__main__":
    main()

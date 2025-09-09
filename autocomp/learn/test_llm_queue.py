#!/usr/bin/env python3
"""
Example usage of the LLM queue system in the autocomp project.
This version waits for results and prints them out.
"""

import time
from autocomp.common.llm_queue import submit_simple_request, LLMRequestWriter

def wait_for_response(writer, request_id, timeout=300):
    """
    Wait for a request to complete and return the response.
    
    Args:
        writer: LLMRequestWriter instance
        request_id: ID of the request to wait for
        timeout: Maximum time to wait in seconds (default: 5 minutes)
    
    Returns:
        LLMResponse object or None if timeout/failed
    """
    start_time = time.time()
    
    print(f"⏳ Waiting for request {request_id[:8]}... to complete...")
    
    while time.time() - start_time < timeout:
        status = writer.get_request_status(request_id)
        
        if status == "completed":
            response = writer.get_response(request_id)
            print(f"✅ Request completed in {time.time() - start_time:.1f}s")
            return response
        elif status == "failed":
            response = writer.get_response(request_id)
            print(f"❌ Request failed in {time.time() - start_time:.1f}s")
            if response and response.error:
                print(f"   Error: {response.error}")
            return response
        elif status == "processing":
            print(f"🔄 Request is being processed... ({time.time() - start_time:.1f}s elapsed)")
        else:
            print(f"📝 Request status: {status} ({time.time() - start_time:.1f}s elapsed)")
        
        time.sleep(2)  # Check every 2 seconds
    
    print(f"⏰ Timeout after {timeout}s - request may still be processing")
    return None

def print_response(response, request_description):
    """Print a formatted response."""
    print(f"\n{'='*60}")
    print(f"📄 {request_description}")
    print(f"{'='*60}")
    
    if response:
        if response.error:
            print(f"❌ Error: {response.error}")
        else:
            print(f"🤖 Model: {response.model}")
            print(f"⏱️  Processing time: {response.processing_time:.2f}s")
            print(f"📅 Created: {response.created_at}")
            print(f"\n📝 Response:")
            print("-" * 40)
            print(response.response)
            print("-" * 40)
    else:
        print("❌ No response received")

def main():
    """Demonstrate basic usage of the LLM queue system with result waiting."""
    
    print("🚀 LLM Queue System Test - With Result Waiting")
    print("=" * 60)
    
    queue_dir = "/nscratch/charleshong/autocomp/llm_queue"
    writer = LLMRequestWriter(queue_dir)
    
    print(f"📁 Queue directory: {queue_dir}")
    print("💡 Make sure the poller is running: python examples/llm_queue_poller.py")
    print()
    
    # Example 1: Submit a simple request
    print("1️⃣ Submitting simple optimization request...")
    
    model_name = "/nscratch/charleshong/autocomp/model/"

    request_id = submit_simple_request(
        queue_dir=queue_dir,
        prompt="Hello, how are you?",
        model=model_name,
        # model="openai/gpt-oss-20b",
        temperature=0.6,
    )
    
    print(f"   📤 Submitted request: {request_id}")
    
    # Wait for and print the response
    response = wait_for_response(writer, request_id)
    print_response(response, "Simple Optimization Request")
    
    # Example 2: Submit a more complex request
    print("\n2️⃣ Submitting complex Gemmini optimization request...")
    messages = [
        {"role": "system", "content": "You are an expert in hardware acceleration and compiler optimization."},
        {"role": "user", "content": "I have a matrix multiplication kernel that needs to be optimized for a Gemmini accelerator. Can you help me analyze the memory access patterns and suggest optimizations?"}
    ]
    
    complex_request_id = writer.submit_request(
        messages=messages,
        model=model_name,  # Use the same model from API
        temperature=0.6,  # Lower temperature for more focused code optimization
        # max_tokens=1000,
    )
    
    print(f"   📤 Submitted complex request: {complex_request_id}")
    
    # Wait for and print the response
    response = wait_for_response(writer, complex_request_id)
    print_response(response, "Complex Gemmini Optimization Request")
    
    print(f"\n🎉 All requests processed!")
    print(f"💾 Results saved in: {queue_dir}/completed/")

if __name__ == "__main__":
    main()

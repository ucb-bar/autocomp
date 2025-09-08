# LLM Queue System Examples

This directory contains examples demonstrating how to use the LLM queue system for writing requests to a directory and processing them with a vLLM inference endpoint.

## Overview

The LLM queue system consists of:
1. **Request Writer** - Writes LLM requests to a directory queue
2. **Queue Poller** - Polls the directory and processes requests through vLLM
3. **vLLM Client** - Handles communication with the vLLM inference server

## Quick Start

### 1. Install Dependencies

Make sure you have the required dependencies:

```bash
pip install aiohttp requests
```

For vLLM server:
```bash
pip install vllm
# or for CUDA support:
pip install vllm[cuda]
```

### 2. Start vLLM Server

Start a vLLM server to handle inference requests:

```bash
# Option 1: Use the helper script
python examples/start_vllm_server.py --model microsoft/DialoGPT-medium --wait

# Option 2: Start manually
python -m vllm.entrypoints.openai.api_server \
    --model microsoft/DialoGPT-medium \
    --host localhost \
    --port 8000
```

### 3. Submit Requests

In another terminal, submit some requests to the queue:

```bash
python examples/llm_queue_writer.py
```

### 4. Start the Poller

In a third terminal, start the poller to process requests:

```bash
python examples/llm_queue_poller.py
```

## File Structure

When you run the examples, the following directory structure will be created:

```
llm_queue/
├── pending/          # New requests waiting to be processed
├── processing/       # Requests currently being processed
├── completed/        # Successfully completed requests
│   ├── {request_id}.json          # Original request
│   └── {request_id}_response.json # Response from vLLM
└── failed/           # Failed requests
    ├── {request_id}.json          # Original request
    └── {request_id}_response.json # Error details
```

## Example Scripts

### llm_queue_writer.py

Demonstrates how to:
- Submit simple text prompts
- Submit multiple requests in batch
- Create multi-turn conversations
- Monitor request status and retrieve responses

### llm_queue_poller.py

Shows how to:
- Set up and run the queue poller
- Configure vLLM endpoint and polling parameters
- Handle concurrent request processing

### start_vllm_server.py

Helper script to:
- Start a vLLM server with common configurations
- Wait for server to be ready
- Provide useful server information

## Usage Examples

### Basic Usage

```python
from autocomp.common.llm_queue import LLMRequestWriter, submit_simple_request

# Submit a simple request
request_id = submit_simple_request(
    queue_dir="./my_queue",
    prompt="Explain machine learning in simple terms",
    model="llama3.1-8b"
)

# Check status
writer = LLMRequestWriter("./my_queue")
status = writer.get_request_status(request_id)
print(f"Request status: {status}")

# Get response (when completed)
response = writer.get_response(request_id)
if response:
    print(f"Response: {response.response}")
```

### Advanced Usage

```python
from autocomp.common.llm_queue import LLMRequestWriter, LLMQueuePoller

# Create queue system
writer = LLMRequestWriter("./my_queue")
poller = LLMQueuePoller("./my_queue", vllm_url="http://localhost:8000")

# Submit complex request
messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Write a Python function to sort a list."}
]

request_id = writer.submit_request(
    messages=messages,
    model="codellama-7b",
    temperature=0.1,
    max_tokens=500
)

# Run poller (in separate process/thread)
# poller.run()
```

## Configuration Options

### vLLM Server Configuration

- `--model`: Model to serve (e.g., `microsoft/DialoGPT-medium`, `codellama/CodeLlama-7b-hf`)
- `--host`: Host to bind to (default: `localhost`)
- `--port`: Port to bind to (default: `8000`)
- `--gpu-memory-utilization`: GPU memory utilization (default: `0.8`)
- `--max-model-len`: Maximum model length

### Queue Poller Configuration

- `--queue-dir`: Directory to poll for requests
- `--vllm-url`: vLLM server URL
- `--poll-interval`: Polling interval in seconds
- `--max-workers`: Maximum concurrent workers

## Tips

1. **Model Selection**: Choose appropriate models for your use case:
   - `microsoft/DialoGPT-medium`: Good for testing, small size
   - `codellama/CodeLlama-7b-hf`: Good for code generation
   - `meta-llama/Llama-2-7b-chat-hf`: Good for general chat

2. **Performance Tuning**:
   - Adjust `--max-workers` based on your GPU memory
   - Use shorter `--poll-interval` for faster response times
   - Monitor GPU utilization and adjust accordingly

3. **Error Handling**:
   - Check the `failed/` directory for error details
   - Monitor vLLM server logs for issues
   - Implement retry logic for failed requests if needed

4. **Scaling**:
   - Run multiple pollers for higher throughput
   - Use different queue directories for different models
   - Implement load balancing across multiple vLLM servers

## Troubleshooting

### Common Issues

1. **vLLM server not starting**:
   - Check GPU memory availability
   - Verify model name and availability
   - Check CUDA installation for GPU models

2. **Requests stuck in processing**:
   - Check vLLM server health
   - Verify network connectivity
   - Check server logs for errors

3. **High memory usage**:
   - Reduce `--max-workers`
   - Lower `--gpu-memory-utilization`
   - Use smaller models for testing

### Getting Help

- Check vLLM documentation: https://docs.vllm.ai/
- Review server logs for detailed error messages
- Monitor system resources (GPU memory, CPU, disk space)

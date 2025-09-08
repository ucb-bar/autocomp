"""
LLM Request Queue System

This module provides tools for writing LLM requests to a directory and polling
that directory to process requests through a vLLM inference endpoint.
"""

import os
import json
import time
import uuid
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import queue
import signal
import sys

from openai import AsyncOpenAI
from autocomp.common import logger


@dataclass
class LLMRequest:
    """Represents an LLM inference request."""
    id: str
    messages: List[Dict[str, str]]
    model: str = "default"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    stream: bool = False
    created_at: str = ""
    status: str = "pending"  # pending, processing, completed, error
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class LLMResponse:
    """Represents an LLM inference response."""
    request_id: str
    response: str
    model: str
    created_at: str = ""
    processing_time: float = 0.0
    error: Optional[str] = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class LLMRequestWriter:
    """Tool for writing LLM requests to a directory."""
    
    def __init__(self, queue_dir: Union[str, Path]):
        """
        Initialize the request writer.
        
        Args:
            queue_dir: Directory where request files will be written
        """
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organization
        self.pending_dir = self.queue_dir / "pending"
        self.processing_dir = self.queue_dir / "processing"
        self.completed_dir = self.queue_dir / "completed"
        self.failed_dir = self.queue_dir / "failed"
        
        for dir_path in [self.pending_dir, self.processing_dir, self.completed_dir, self.failed_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def create_request(
        self,
        messages: List[Dict[str, str]],
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMRequest:
        """
        Create an LLM request object.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: Model name to use for inference
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters for the request
            
        Returns:
            LLMRequest object
        """
        request_id = str(uuid.uuid4())
        
        request = LLMRequest(
            id=request_id,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return request
    
    def write_request(self, request: LLMRequest) -> Path:
        """
        Write a request to the pending queue.
        
        Args:
            request: LLMRequest object to write
            
        Returns:
            Path to the written request file
        """
        request_file = self.pending_dir / f"{request.id}.json"
        
        with open(request_file, 'w') as f:
            json.dump(asdict(request), f, indent=2)
        
        logger.info(f"Written request {request.id} to {request_file}")
        return request_file
    
    def submit_request(
        self,
        messages: List[Dict[str, str]],
        model: str = "default",
        **kwargs
    ) -> str:
        """
        Create and submit a request in one step.
        
        Args:
            messages: List of message dictionaries
            model: Model name
            **kwargs: Additional request parameters
            
        Returns:
            Request ID
        """
        request = self.create_request(messages, model, **kwargs)
        self.write_request(request)
        return request.id
    
    def get_request_status(self, request_id: str) -> Optional[str]:
        """
        Get the status of a request.
        
        Args:
            request_id: ID of the request
            
        Returns:
            Status string or None if not found
        """
        # Check all directories for the request
        for status_dir, status in [
            (self.pending_dir, "pending"),
            (self.processing_dir, "processing"),
            (self.completed_dir, "completed"),
            (self.failed_dir, "failed")
        ]:
            request_file = status_dir / f"{request_id}.json"
            if request_file.exists():
                return status
        
        return None
    
    def get_response(self, request_id: str) -> Optional[LLMResponse]:
        """
        Get the response for a completed request.
        
        Args:
            request_id: ID of the request
            
        Returns:
            LLMResponse object or None if not found/completed
        """
        response_file = self.completed_dir / f"{request_id}_response.json"
        
        if not response_file.exists():
            return None
        
        try:
            with open(response_file, 'r') as f:
                response_data = json.load(f)
            return LLMResponse(**response_data)
        except Exception as e:
            logger.error(f"Error loading response for {request_id}: {e}")
            return None


class VLLMClient:
    """Client for communicating with vLLM inference endpoint using AsyncOpenAI."""
    
    def __init__(self, base_url: str = "http://localhost:8000", enable_batching: bool = True, max_batch_size: int = 32):
        """
        Initialize vLLM client using AsyncOpenAI.
        
        Args:
            base_url: Base URL of the vLLM server
            enable_batching: Whether to enable request batching for better throughput
            max_batch_size: Maximum number of requests to batch together
        """
        self.base_url = base_url.rstrip('/')
        self.enable_batching = enable_batching
        self.max_batch_size = max_batch_size
        
        # Initialize AsyncOpenAI client for vLLM
        self.client = AsyncOpenAI(
            api_key="EMPTY",  # vLLM doesn't require a real API key
            base_url=f"{self.base_url}/v1"
        )
        
        # Semaphore for controlling concurrent requests
        self.semaphore = asyncio.Semaphore(max_batch_size)
    
    async def check_health(self) -> bool:
        """Check if vLLM server is healthy."""
        try:
            models = await self.client.models.list()
            return len(models.data) > 0
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def get_models(self) -> List[str]:
        """Get available models from vLLM server."""
        try:
            models = await self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            return []
    
    async def generate_completion(self, request: LLMRequest) -> LLMResponse:
        """
        Generate completion using AsyncOpenAI client.
        
        Args:
            request: LLMRequest object
            
        Returns:
            LLMResponse object
        """
        results = await self.generate_completions([request])
        return results[0]
    
    async def generate_completions(self, requests: List[LLMRequest]) -> List[LLMResponse]:
        """
        Generate completions for multiple requests using AsyncOpenAI with concurrent processing.
        
        Args:
            requests: List of LLMRequest objects
            
        Returns:
            List of LLMResponse objects in the same order as input requests
        """
        if not requests:
            return []
        
        # Process all requests concurrently using AsyncOpenAI
        tasks = [self._process_single_request(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error responses
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing request {requests[i].id}: {result}")
                processed_results.append(LLMResponse(
                    request_id=requests[i].id,
                    response="",
                    model=requests[i].model,
                    processing_time=0.0,
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _process_single_request(self, request: LLMRequest) -> LLMResponse:
        """Process a single request using AsyncOpenAI with retry logic."""
        start_time = time.time()
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                async with self.semaphore:  # Control concurrency
                    # Prepare request parameters
                    kwargs = {
                        "model": request.model,
                        "messages": request.messages,
                        "temperature": request.temperature,
                        "stream": request.stream,
                    }
                    
                    # Add optional parameters if specified
                    if request.max_tokens:
                        kwargs["max_tokens"] = request.max_tokens
                    if request.top_p != 1.0:
                        kwargs["top_p"] = request.top_p
                    if request.frequency_penalty != 0.0:
                        kwargs["frequency_penalty"] = request.frequency_penalty
                    if request.presence_penalty != 0.0:
                        kwargs["presence_penalty"] = request.presence_penalty
                    if request.stop:
                        kwargs["stop"] = request.stop
                    
                    # Make the API call
                    response = await self.client.chat.completions.create(**kwargs)
                    
                    # Extract response content
                    content = response.choices[0].message.content
                    processing_time = time.time() - start_time
                    
                    return LLMResponse(
                        request_id=request.id,
                        response=content,
                        model=request.model,
                        processing_time=processing_time
                    )
            
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for request {request.id}: {e}")
                if attempt == max_retries - 1:  # Last attempt
                    return LLMResponse(
                        request_id=request.id,
                        response="",
                        model=request.model,
                        processing_time=time.time() - start_time,
                        error=str(e)
                    )
                
                # Exponential backoff for retries
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
        
        # This should never be reached, but just in case
        return LLMResponse(
            request_id=request.id,
            response="",
            model=request.model,
            processing_time=time.time() - start_time,
            error="Max retries exceeded"
        )


class LLMQueuePoller:
    """Tool for polling a directory and processing LLM requests."""
    
    def __init__(
        self,
        queue_dir: Union[str, Path],
        vllm_url: str = "http://localhost:8000",
        poll_interval: float = 1.0,
        max_workers: int = 4,
        enable_batching: bool = True,
        max_batch_size: int = 32,
        batch_timeout: float = 2.0
    ):
        """
        Initialize the queue poller.
        
        Args:
            queue_dir: Directory to poll for requests
            vllm_url: URL of the vLLM server
            poll_interval: Polling interval in seconds
            max_workers: Maximum number of concurrent workers
            enable_batching: Whether to enable request batching
            max_batch_size: Maximum requests per batch
            batch_timeout: Maximum time to wait for batch to fill up
        """
        self.queue_dir = Path(queue_dir)
        self.vllm_client = VLLMClient(vllm_url, enable_batching, max_batch_size)
        self.poll_interval = poll_interval
        self.max_workers = max_workers
        self.enable_batching = enable_batching
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        
        # Directory structure
        self.pending_dir = self.queue_dir / "pending"
        self.processing_dir = self.queue_dir / "processing"
        self.completed_dir = self.queue_dir / "completed"
        self.failed_dir = self.queue_dir / "failed"
        
        # Create directories if they don't exist
        for dir_path in [self.pending_dir, self.processing_dir, self.completed_dir, self.failed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Control variables
        self.running = False
        self.worker_semaphore = asyncio.Semaphore(max_workers)
        
        # Batching control
        self.pending_batches = {}  # model -> list of requests
        self.batch_lock = asyncio.Lock()
        
        # Performance tracking
        self.stats = {
            'requests_processed': 0,
            'batches_processed': 0,
            'total_processing_time': 0.0,
            'requests_per_batch': []
        }
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    async def _process_request(self, request_file: Path):
        """Process a single request file."""
        async with self.worker_semaphore:
            try:
                # Load request
                with open(request_file, 'r') as f:
                    request_data = json.load(f)
                
                request = LLMRequest(**request_data)
                logger.info(f"Processing request {request.id}")
                
                # Move to processing directory
                processing_file = self.processing_dir / request_file.name
                request_file.rename(processing_file)
                
                # Update status
                request.status = "processing"
                with open(processing_file, 'w') as f:
                    json.dump(asdict(request), f, indent=2)
                
                # Generate completion
                response = await self.vllm_client.generate_completion(request)
                
                if response.error:
                    # Move to failed directory
                    failed_file = self.failed_dir / request_file.name
                    processing_file.rename(failed_file)
                    
                    # Update request status
                    request.status = "failed"
                    with open(failed_file, 'w') as f:
                        json.dump(asdict(request), f, indent=2)
                    
                    # Save error response
                    error_response_file = self.failed_dir / f"{request.id}_response.json"
                    with open(error_response_file, 'w') as f:
                        json.dump(asdict(response), f, indent=2)
                    
                    logger.error(f"Request {request.id} failed: {response.error}")
                
                else:
                    # Move to completed directory
                    completed_file = self.completed_dir / request_file.name
                    processing_file.rename(completed_file)
                    
                    # Update request status
                    request.status = "completed"
                    with open(completed_file, 'w') as f:
                        json.dump(asdict(request), f, indent=2)
                    
                    # Save response
                    response_file = self.completed_dir / f"{request.id}_response.json"
                    with open(response_file, 'w') as f:
                        json.dump(asdict(response), f, indent=2)
                    
                    logger.info(f"Request {request.id} completed successfully in {response.processing_time:.2f}s")
            
            except Exception as e:
                logger.error(f"Error processing request {request_file}: {e}")
                # Move to failed directory if still in processing
                if request_file.parent == self.processing_dir:
                    failed_file = self.failed_dir / request_file.name
                    request_file.rename(failed_file)
    
    async def _poll_once(self):
        """Poll the pending directory once for new requests."""
        try:
            pending_files = list(self.pending_dir.glob("*.json"))
            
            if pending_files:
                logger.info(f"Found {len(pending_files)} pending requests")
                
                if self.enable_batching and len(pending_files) > 1:
                    await self._process_requests_with_batching(pending_files)
                else:
                    # Process requests individually (original behavior)
                    tasks = [self._process_request(request_file) for request_file in pending_files]
                    await asyncio.gather(*tasks, return_exceptions=True)
        
        except Exception as e:
            logger.error(f"Error during polling: {e}")
    
    async def _process_requests_with_batching(self, request_files: List[Path]):
        """Process requests using AsyncOpenAI-based batching."""
        # Load all requests
        requests_and_files = []
        
        for request_file in request_files:
            try:
                with open(request_file, 'r') as f:
                    request_data = json.load(f)
                
                request = LLMRequest(**request_data)
                requests_and_files.append((request, request_file))
                
            except Exception as e:
                logger.error(f"Error loading request {request_file}: {e}")
                # Move problematic file to failed directory
                failed_file = self.failed_dir / request_file.name
                request_file.rename(failed_file)
        
        if requests_and_files:
            # Process all requests as a single batch using AsyncOpenAI's built-in concurrency
            await self._process_request_batch(requests_and_files)
    
    async def _process_request_batch(self, requests_and_files: List[tuple]):
        """Process a batch of requests using AsyncOpenAI."""
        async with self.worker_semaphore:
            try:
                requests = [req for req, _ in requests_and_files]
                request_files = [file for _, file in requests_and_files]
                
                logger.info(f"Processing batch of {len(requests)} requests")
                
                # Move all files to processing directory and update status
                processing_files = []
                for request, request_file in requests_and_files:
                    processing_file = self.processing_dir / request_file.name
                    request_file.rename(processing_file)
                    processing_files.append(processing_file)
                    
                    # Update request status
                    request.status = "processing"
                    with open(processing_file, 'w') as f:
                        json.dump(asdict(request), f, indent=2)
                
                # Generate completions using AsyncOpenAI (handles concurrency internally)
                start_time = time.time()
                responses = await self.vllm_client.generate_completions(requests)
                processing_time = time.time() - start_time
                
                # Update statistics
                self.stats['requests_processed'] += len(requests)
                self.stats['batches_processed'] += 1
                self.stats['total_processing_time'] += processing_time
                self.stats['requests_per_batch'].append(len(requests))
                
                # Process each response
                for i, (request, response) in enumerate(zip(requests, responses)):
                    processing_file = processing_files[i]
                    
                    if response.error:
                        # Move to failed directory
                        failed_file = self.failed_dir / processing_file.name
                        processing_file.rename(failed_file)
                        
                        # Update request status
                        request.status = "failed"
                        with open(failed_file, 'w') as f:
                            json.dump(asdict(request), f, indent=2)
                        
                        # Save error response
                        error_response_file = self.failed_dir / f"{request.id}_response.json"
                        with open(error_response_file, 'w') as f:
                            json.dump(asdict(response), f, indent=2)
                        
                        logger.error(f"Request {request.id} failed in batch: {response.error}")
                    
                    else:
                        # Move to completed directory
                        completed_file = self.completed_dir / processing_file.name
                        processing_file.rename(completed_file)
                        
                        # Update request status
                        request.status = "completed"
                        with open(completed_file, 'w') as f:
                            json.dump(asdict(request), f, indent=2)
                        
                        # Save response
                        response_file = self.completed_dir / f"{request.id}_response.json"
                        with open(response_file, 'w') as f:
                            json.dump(asdict(response), f, indent=2)
                        
                        logger.info(f"Request {request.id} completed in batch ({response.processing_time:.2f}s)")
                
                logger.info(f"Batch of {len(requests)} requests completed in {processing_time:.2f}s")
            
            except Exception as e:
                logger.error(f"Error processing request batch: {e}")
                # Move any remaining processing files to failed
                for processing_file in processing_files:
                    if processing_file.exists():
                        failed_file = self.failed_dir / processing_file.name
                        processing_file.rename(failed_file)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if self.stats['requests_processed'] == 0:
            return self.stats
        
        avg_processing_time = self.stats['total_processing_time'] / self.stats['batches_processed'] if self.stats['batches_processed'] > 0 else 0
        avg_batch_size = sum(self.stats['requests_per_batch']) / len(self.stats['requests_per_batch']) if self.stats['requests_per_batch'] else 0
        
        return {
            **self.stats,
            'avg_processing_time_per_batch': avg_processing_time,
            'avg_batch_size': avg_batch_size,
            'throughput_requests_per_second': self.stats['requests_processed'] / self.stats['total_processing_time'] if self.stats['total_processing_time'] > 0 else 0
        }
    
    async def start_polling(self):
        """Start the polling loop."""
        logger.info(f"Starting LLM queue poller on {self.queue_dir}")
        logger.info(f"vLLM endpoint: {self.vllm_client.base_url}")
        logger.info(f"Poll interval: {self.poll_interval}s, Max workers: {self.max_workers}")
        logger.info(f"Batching enabled: {self.enable_batching}")
        if self.enable_batching:
            logger.info(f"Max batch size: {self.max_batch_size}, Batch timeout: {self.batch_timeout}s")
        
        # Check vLLM health
        if not await self.vllm_client.check_health():
            logger.warning("vLLM server health check failed, continuing anyway...")
        else:
            models = await self.vllm_client.get_models()
            logger.info(f"Available models: {models}")
        
        self.running = True
        last_stats_time = time.time()
        
        try:
            while self.running:
                await self._poll_once()
                
                # Print performance stats every 60 seconds
                current_time = time.time()
                if current_time - last_stats_time >= 60:
                    stats = self.get_performance_stats()
                    if stats['requests_processed'] > 0:
                        logger.info(f"Performance stats: {stats['requests_processed']} requests processed, "
                                  f"{stats['batches_processed']} batches, "
                                  f"avg batch size: {stats.get('avg_batch_size', 0):.1f}, "
                                  f"throughput: {stats.get('throughput_requests_per_second', 0):.2f} req/s")
                    last_stats_time = current_time
                
                await asyncio.sleep(self.poll_interval)
        
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        
        finally:
            self.running = False
            # Print final stats
            final_stats = self.get_performance_stats()
            if final_stats['requests_processed'] > 0:
                logger.info(f"Final performance stats: {final_stats}")
            logger.info("Poller stopped")
    
    def run(self):
        """Run the poller (blocking)."""
        try:
            asyncio.run(self.start_polling())
        except KeyboardInterrupt:
            logger.info("Shutting down...")


# Convenience functions
def create_queue_system(queue_dir: Union[str, Path]) -> tuple[LLMRequestWriter, LLMQueuePoller]:
    """
    Create a complete queue system with writer and poller.
    
    Args:
        queue_dir: Directory for the queue
        
    Returns:
        Tuple of (LLMRequestWriter, LLMQueuePoller)
    """
    writer = LLMRequestWriter(queue_dir)
    poller = LLMQueuePoller(queue_dir)
    return writer, poller


def submit_simple_request(
    queue_dir: Union[str, Path],
    prompt: str,
    model: str = "default",
    **kwargs
) -> str:
    """
    Submit a simple text prompt as a request.
    
    Args:
        queue_dir: Queue directory
        prompt: Text prompt
        model: Model name
        **kwargs: Additional request parameters
        
    Returns:
        Request ID
    """
    writer = LLMRequestWriter(queue_dir)
    messages = [{"role": "user", "content": prompt}]
    return writer.submit_request(messages, model, **kwargs)

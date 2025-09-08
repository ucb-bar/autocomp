#!/usr/bin/env python3
"""
Command-line interface for managing the LLM queue system.

This tool provides easy commands for:
- Submitting requests
- Checking queue status
- Retrieving responses
- Managing the queue
"""

import argparse
import json
import sys
from pathlib import Path
from autocomp.common.llm_queue import LLMRequestWriter, submit_simple_request


def submit_command(args):
    """Handle submit command."""
    if args.file:
        # Read messages from file
        with open(args.file, 'r') as f:
            data = json.load(f)
            messages = data.get('messages', [])
            model = data.get('model', args.model)
            temperature = data.get('temperature', args.temperature)
            max_tokens = data.get('max_tokens', args.max_tokens)
    else:
        # Simple prompt
        messages = [{"role": "user", "content": args.prompt}]
        model = args.model
        temperature = args.temperature
        max_tokens = args.max_tokens
    
    writer = LLMRequestWriter(args.queue_dir)
    
    request_id = writer.submit_request(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    print(f"Submitted request: {request_id}")
    return request_id


def status_command(args):
    """Handle status command."""
    writer = LLMRequestWriter(args.queue_dir)
    
    if args.request_id:
        # Check specific request
        status = writer.get_request_status(args.request_id)
        if status:
            print(f"Request {args.request_id}: {status}")
            
            if status == "completed":
                response = writer.get_response(args.request_id)
                if response:
                    print(f"Processing time: {response.processing_time:.2f}s")
                    if args.verbose:
                        print(f"Response: {response.response}")
            elif status == "failed":
                response = writer.get_response(args.request_id)
                if response and response.error:
                    print(f"Error: {response.error}")
        else:
            print(f"Request {args.request_id} not found")
    else:
        # Show queue overview
        queue_dir = Path(args.queue_dir)
        
        pending = len(list((queue_dir / "pending").glob("*.json")))
        processing = len(list((queue_dir / "processing").glob("*.json")))
        completed = len(list((queue_dir / "completed").glob("*.json")))
        failed = len(list((queue_dir / "failed").glob("*.json")))
        
        print("Queue Status:")
        print(f"  Pending:    {pending}")
        print(f"  Processing: {processing}")
        print(f"  Completed:  {completed}")
        print(f"  Failed:     {failed}")
        print(f"  Total:      {pending + processing + completed + failed}")


def list_command(args):
    """Handle list command."""
    queue_dir = Path(args.queue_dir)
    status_dir = queue_dir / args.status
    
    if not status_dir.exists():
        print(f"Status directory {status_dir} does not exist")
        return
    
    files = list(status_dir.glob("*.json"))
    files = [f for f in files if not f.name.endswith("_response.json")]
    
    print(f"{args.status.title()} requests ({len(files)}):")
    
    for request_file in sorted(files):
        try:
            with open(request_file, 'r') as f:
                data = json.load(f)
            
            request_id = data['id']
            created_at = data['created_at']
            model = data.get('model', 'unknown')
            
            # Get first user message as preview
            user_messages = [msg for msg in data['messages'] if msg['role'] == 'user']
            preview = user_messages[0]['content'][:50] + "..." if user_messages else "No user message"
            
            print(f"  {request_id[:8]}... | {created_at} | {model} | {preview}")
            
        except Exception as e:
            print(f"  Error reading {request_file.name}: {e}")


def get_command(args):
    """Handle get command."""
    writer = LLMRequestWriter(args.queue_dir)
    response = writer.get_response(args.request_id)
    
    if response:
        if args.json:
            print(json.dumps({
                'request_id': response.request_id,
                'response': response.response,
                'model': response.model,
                'created_at': response.created_at,
                'processing_time': response.processing_time,
                'error': response.error
            }, indent=2))
        else:
            print(f"Request ID: {response.request_id}")
            print(f"Model: {response.model}")
            print(f"Processing Time: {response.processing_time:.2f}s")
            print(f"Created: {response.created_at}")
            if response.error:
                print(f"Error: {response.error}")
            else:
                print("Response:")
                print("-" * 40)
                print(response.response)
                print("-" * 40)
    else:
        status = writer.get_request_status(args.request_id)
        if status:
            print(f"Request {args.request_id} is {status}, no response available yet")
        else:
            print(f"Request {args.request_id} not found")


def clean_command(args):
    """Handle clean command."""
    queue_dir = Path(args.queue_dir)
    
    if args.status == "all":
        dirs_to_clean = ["pending", "processing", "completed", "failed"]
    else:
        dirs_to_clean = [args.status]
    
    total_removed = 0
    
    for status in dirs_to_clean:
        status_dir = queue_dir / status
        if status_dir.exists():
            files = list(status_dir.glob("*.json"))
            for file in files:
                file.unlink()
                total_removed += 1
    
    print(f"Removed {total_removed} files")


def main():
    parser = argparse.ArgumentParser(description="LLM Queue CLI Tool")
    parser.add_argument(
        "--queue-dir",
        type=str,
        default="./llm_queue",
        help="Queue directory (default: ./llm_queue)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit a new request")
    submit_group = submit_parser.add_mutually_exclusive_group(required=True)
    submit_group.add_argument("--prompt", type=str, help="Text prompt to submit")
    submit_group.add_argument("--file", type=str, help="JSON file with request data")
    
    submit_parser.add_argument("--model", type=str, default="default", help="Model name")
    submit_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    submit_parser.add_argument("--max-tokens", type=int, help="Maximum tokens")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check queue or request status")
    status_parser.add_argument("--request-id", type=str, help="Specific request ID to check")
    status_parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed info")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List requests by status")
    list_parser.add_argument(
        "status",
        choices=["pending", "processing", "completed", "failed"],
        help="Status to list"
    )
    
    # Get command
    get_parser = subparsers.add_parser("get", help="Get response for a request")
    get_parser.add_argument("request_id", help="Request ID")
    get_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean queue directories")
    clean_parser.add_argument(
        "status",
        choices=["pending", "processing", "completed", "failed", "all"],
        help="Status directory to clean"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == "submit":
            submit_command(args)
        elif args.command == "status":
            status_command(args)
        elif args.command == "list":
            list_command(args)
        elif args.command == "get":
            get_command(args)
        elif args.command == "clean":
            clean_command(args)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

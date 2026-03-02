import os
import re
import json
import asyncio
import random
import copy

import backoff
import boto3
import openai
from openai import OpenAI, AsyncOpenAI, RateLimitError, APITimeoutError, InternalServerError
from google import genai
from google.genai import types
import anthropic
from anthropic import AsyncAnthropic
from together import Together, AsyncTogether
from mistralai_gcp import MistralGoogleCloud

from autocomp.common import logger

# Try to import keys from keys.py, use as fallback if env vars not set
try:
    from autocomp.common import keys
except ImportError:
    keys = None

def _get_key(env_var: str, default: str = "EMPTY"):
    """Get key from environment variable, falling back to keys.py."""
    value = os.environ.get(env_var)
    if value:
        return value
    if keys and hasattr(keys, env_var):
        file_value = getattr(keys, env_var)
        if file_value is not None:
            return file_value
    return default

openai_key_str = _get_key("OPENAI_API_KEY")
anthropic_key_str = _get_key("ANTHROPIC_API_KEY")
together_key_str = _get_key("TOGETHER_API_KEY")
aws_access_key = _get_key("AWS_ACCESS_KEY_ID", default=None)
aws_secret_key = _get_key("AWS_SECRET_ACCESS_KEY", default=None)
aws_region = _get_key("AWS_REGION", default="us-west-2")
google_cloud_project = _get_key("GOOGLE_CLOUD_PROJECT", default=None)
google_cloud_location = _get_key("GOOGLE_CLOUD_LOCATION", default=None)
vllm_api_base = _get_key("VLLM_API_BASE", default="http://localhost:8000/v1")

# Log key availability
_key_status = {
    "OPENAI_API_KEY": openai_key_str not in (None, "EMPTY"),
    "ANTHROPIC_API_KEY": anthropic_key_str not in (None, "EMPTY"),
    "TOGETHER_API_KEY": together_key_str not in (None, "EMPTY"),
    "AWS_ACCESS_KEY_ID": aws_access_key is not None,
    "AWS_SECRET_ACCESS_KEY": aws_secret_key is not None,
    "GOOGLE_CLOUD_PROJECT": google_cloud_project is not None,
    "GOOGLE_CLOUD_LOCATION": google_cloud_location is not None,
}
_available = [k for k, v in _key_status.items() if v]
_unavailable = [k for k, v in _key_status.items() if not v]
if _available:
    logger.info(f"Keys available: {', '.join(_available)}")
if _unavailable:
    logger.info(f"Keys unavailable: {', '.join(_unavailable)}")

def is_openai_reasoning_model(model: str) -> bool:
    return "o1" in model or "o3" in model or "o4" in model or "gpt-5" in model
def can_web_search_openai(model: str) -> bool:
    return "gpt-5" in model or model in ["o4-mini", "o4"]

def extract(s):
    # return [x for x in re.findall(r"```(?:python|Python)?(.*)```", s, re.DOTALL)]
    return [x for x in re.findall(r"```(?:c|c\+\+|cpp)?\n(void test\(.*}\n.*)```", s, re.DOTALL)]

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def completions_with_backoff(client: OpenAI | Together, **kwargs):
    if isinstance(client, Together):
        return client.chat.completions.create(**kwargs)
    else:
        return client.responses.create(**kwargs)

async def fetch_completion(semaphore: asyncio.Semaphore, client: AsyncOpenAI | AsyncTogether | genai.Client | AsyncAnthropic, prompt, **kwargs):
    """Fetches a completion with retries and rate limit handling."""
    max_retries = 8
    for attempt in range(max_retries):
        try:
            async with semaphore:  # Limits concurrent requests
                if isinstance(client, AsyncTogether):
                    # Together uses chat completions API with messages format
                    messages = [{"role": "user", "content": prompt}]
                    response = await client.chat.completions.create(messages=messages, **kwargs)
                elif isinstance(client, AsyncOpenAI):
                    # OpenAI uses responses API
                    response = await client.responses.create(input=prompt, **kwargs)
                elif isinstance(client, genai.Client):
                    response = await client.aio.models.generate_content(
                        model=kwargs["model"],
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=kwargs["temperature"],
                            candidate_count=kwargs["n"],
                        ),
                    )
                elif isinstance(client, AsyncAnthropic) or isinstance(client, anthropic.AsyncAnthropicBedrock):
                    # Anthropic uses messages API
                    messages = [{"role": "user", "content": prompt}]
                    anthropic_kwargs = {
                        "model": kwargs["model"],
                        "messages": messages,
                        "max_tokens": kwargs.get("max_tokens", 21000),
                    }
                    if "temperature" in kwargs:
                        anthropic_kwargs["temperature"] = kwargs["temperature"]
                    response = await client.messages.create(**anthropic_kwargs)
            return response
        
        # except (RateLimitError, APITimeoutError, InternalServerError):
        except Exception as e:
            err_str = str(e)
            if "temperature" in err_str and "not supported" in err_str:
                logger.info(f"Model does not support temperature, retrying without it: {e}")
                kwargs.pop("temperature", None)
                continue
            logger.info(f"Error: {e}")
            wait_time = 2 ** attempt + random.uniform(0, 1)  # Exponential backoff
            logger.info(f"Rate limit hit! Retrying in {wait_time:.2f} seconds...")
            await asyncio.sleep(wait_time)
    
    print("Max retries reached, request failed.")
    return None

async def fetch_web_search_completion(semaphore: asyncio.Semaphore, client: AsyncOpenAI | AsyncTogether, query: str, **kwargs):
    """Fetches a web search completion with retries and rate limit handling."""
    max_retries = 8
    for attempt in range(max_retries):
        try:
            async with semaphore:  # Limits concurrent requests
                response = await client.responses.create(
                    model=kwargs["model"],
                    tools=[{"type": "web_search_preview"}],
                    input=query,
                    **{k: v for k, v in kwargs.items() if k != "model"}
                )
            return response
        
        except Exception as e:
            logger.info(f"Web search error: {e}")
            wait_time = 2 ** attempt + random.uniform(0, 1)  # Exponential backoff
            logger.info(f"Rate limit hit! Retrying in {wait_time:.2f} seconds...")
            await asyncio.sleep(wait_time)
    
    print("Max retries reached, web search request failed.")
    return None

async def fetch_web_search_completions(client: AsyncOpenAI | AsyncTogether, queries: list[str], **kwargs) -> list[str]:
    """
    Async version of web search for multiple queries.
    
    Args:
        client: The async client to use (OpenAI or Together)
        queries: List of search queries
        **kwargs: Additional parameters for the API calls
    
    Returns:
        List of web search results as strings
    
    Example:
        queries = [
            "latest Python 3.12 features",
            "best practices for async programming",
            "GPU optimization techniques 2024"
        ]
        
        results = await fetch_web_search_completions(client, queries)
        # returns ["result1", "result2", "result3"]
    """
    MAX_CONCURRENT_REQUESTS = 5  # Lower for web search to be respectful
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    tasks = [fetch_web_search_completion(semaphore, client, query, **kwargs) for query in queries]
    results = await asyncio.gather(*tasks)
    
    responses = []
    for resp in results:
        if resp is None:
            responses.append("Web search failed")
            continue
            
        # Handle OpenAI/Together response
        if hasattr(resp, 'output_text'):
            responses.append(resp.output_text)
        elif hasattr(resp, 'choices') and resp.choices:
            responses.append(resp.choices[0].message.content)
        elif hasattr(resp, 'output') and resp.output:
            responses.append(resp["output"][1]["content"][0]["text"])
        else:
            responses.append("No results found")
    
    return responses

##############################################################################
# Provider-specific translation helpers for tool calling / structured output
##############################################################################

def _openai_tools_from_schema(tools: list[dict]) -> list[dict]:
    """OpenAI/vLLM/Together: pass through as-is."""
    return tools


def _anthropic_tools_from_schema(tools: list[dict]) -> list[dict]:
    """Convert OpenAI tool schema to Anthropic format."""
    out = []
    for t in tools:
        fn = t["function"]
        out.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return out


def _gemini_tools_from_schema(tools: list[dict]) -> list[types.Tool]:
    """Convert OpenAI tool schema to Gemini FunctionDeclaration list."""
    declarations = []
    for t in tools:
        fn = t["function"]
        declarations.append(types.FunctionDeclaration(
            name=fn["name"],
            description=fn.get("description", ""),
            parameters=fn.get("parameters"),
        ))
    return [types.Tool(function_declarations=declarations)]


def _bedrock_tools_from_schema(tools: list[dict]) -> dict:
    """Convert OpenAI tool schema to Bedrock Converse toolConfig."""
    tool_specs = []
    for t in tools:
        fn = t["function"]
        tool_specs.append({
            "toolSpec": {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "inputSchema": {
                    "json": fn.get("parameters", {"type": "object", "properties": {}}),
                },
            }
        })
    return {"tools": tool_specs}


def _normalize_openai_response(message) -> dict:
    """Normalize an OpenAI/vLLM/Together ChatCompletion message to common dict."""
    tool_calls = []
    if message.tool_calls:
        for tc in message.tool_calls:
            tool_calls.append({
                "id": tc.id,
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            })
    return {"role": "assistant", "content": message.content, "tool_calls": tool_calls}


def _normalize_anthropic_response(response) -> dict:
    """Normalize an Anthropic Messages response to common dict."""
    content_text = ""
    tool_calls = []
    for block in response.content:
        if block.type == "text":
            content_text += block.text
        elif block.type == "tool_use":
            tool_calls.append({
                "id": block.id,
                "function": {"name": block.name, "arguments": json.dumps(block.input)},
            })
    return {"role": "assistant", "content": content_text or None, "tool_calls": tool_calls}


def _normalize_gemini_response(response) -> dict:
    """Normalize a Gemini generateContent response to common dict."""
    content_text = ""
    tool_calls = []
    for part in response.candidates[0].content.parts:
        if part.text:
            content_text += part.text
        if part.function_call:
            fc = part.function_call
            tool_calls.append({
                "id": f"gemini_{fc.name}_{id(fc)}",
                "function": {"name": fc.name, "arguments": json.dumps(dict(fc.args) if fc.args else {})},
            })
    return {"role": "assistant", "content": content_text or None, "tool_calls": tool_calls}


def _normalize_bedrock_response(response) -> dict:
    """Normalize a Bedrock Converse response to common dict."""
    content_text = ""
    tool_calls = []
    for block in response["output"]["message"]["content"]:
        if "text" in block:
            content_text += block["text"]
        elif "toolUse" in block:
            tu = block["toolUse"]
            tool_calls.append({
                "id": tu["toolUseId"],
                "function": {"name": tu["name"], "arguments": json.dumps(tu["input"])},
            })
    return {"role": "assistant", "content": content_text or None, "tool_calls": tool_calls}


def _messages_for_anthropic(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Split a messages list into (system_prompt, anthropic_messages).
    Anthropic requires system as a top-level param, not in the messages array.
    Consecutive tool results are grouped into a single user message."""
    system = None
    out = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        elif m["role"] == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": m["tool_call_id"],
                "content": m["content"],
            }
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list) \
                    and out[-1]["content"] and out[-1]["content"][0].get("type") == "tool_result":
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif m["role"] == "assistant" and m.get("tool_calls"):
            content = []
            if m.get("content"):
                content.append({"type": "text", "text": m["content"]})
            for tc in m["tool_calls"]:
                args = tc["function"]["arguments"]
                content.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "input": json.loads(args) if isinstance(args, str) else args,
                })
            out.append({"role": "assistant", "content": content})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return system, out


def _messages_for_gemini(messages: list[dict]) -> tuple[str | None, list]:
    """Convert messages to Gemini format (system_instruction, contents)."""
    # Build a lookup from tool_call_id to function name so FunctionResponse
    # can reference the correct name (Gemini requires it to match).
    tc_id_to_name = {}
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                tc_id_to_name[tc["id"]] = tc["function"]["name"]

    system = None
    contents = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        elif m["role"] == "tool":
            fn_name = tc_id_to_name.get(m.get("tool_call_id"), "tool_result")
            contents.append(types.Content(
                role="user",
                parts=[types.Part(function_response=types.FunctionResponse(
                    name=fn_name,
                    response={"result": m["content"]},
                ))],
            ))
        elif m["role"] == "assistant" and m.get("tool_calls"):
            parts = []
            if m.get("content"):
                parts.append(types.Part(text=m["content"]))
            for tc in m["tool_calls"]:
                args = tc["function"]["arguments"]
                parts.append(types.Part(function_call=types.FunctionCall(
                    name=tc["function"]["name"],
                    args=json.loads(args) if isinstance(args, str) else args,
                )))
            contents.append(types.Content(role="model", parts=parts))
        else:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
    return system, contents


def _messages_for_bedrock(messages: list[dict]) -> tuple[list[dict] | None, list[dict]]:
    """Convert messages to Bedrock Converse format.
    Consecutive tool results are grouped into a single user message."""
    system = None
    out = []
    for m in messages:
        if m["role"] == "system":
            system = [{"text": m["content"]}]
        elif m["role"] == "tool":
            block = {"toolResult": {
                "toolUseId": m["tool_call_id"],
                "content": [{"text": m["content"]}],
            }}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list) \
                    and out[-1]["content"] and "toolResult" in out[-1]["content"][0]:
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif m["role"] == "assistant" and m.get("tool_calls"):
            content = []
            if m.get("content"):
                content.append({"text": m["content"]})
            for tc in m["tool_calls"]:
                args = tc["function"]["arguments"]
                content.append({"toolUse": {
                    "toolUseId": tc["id"],
                    "name": tc["function"]["name"],
                    "input": json.loads(args) if isinstance(args, str) else args,
                }})
            out.append({"role": "assistant", "content": content})
        else:
            out.append({"role": m["role"], "content": [{"text": m["content"]}]})
    return system, out


async def fetch_tool_completion(
    semaphore: asyncio.Semaphore,
    client,
    messages: list[dict],
    provider: str,
    model: str,
    tools: list[dict] | None = None,
    response_format: dict | None = None,
    temperature: float | None = None,
    max_tokens: int = 16384,
    bedrock_client=None,
) -> dict:
    """Provider-dispatching async helper for chat-completions with tool calling.

    Returns a normalized dict: {"role": "assistant", "content": ..., "tool_calls": [...]}.
    """
    max_retries = 8
    for attempt in range(max_retries):
        try:
            async with semaphore:
                if provider in ("openai", "vllm", "together"):
                    kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens}
                    if tools:
                        kwargs["tools"] = _openai_tools_from_schema(tools)
                    if temperature is not None:
                        kwargs["temperature"] = temperature
                    if response_format:
                        kwargs["response_format"] = response_format
                    if isinstance(client, AsyncTogether):
                        resp = await client.chat.completions.create(**kwargs)
                    else:
                        resp = await client.chat.completions.create(**kwargs)
                    return _normalize_openai_response(resp.choices[0].message)

                elif provider in ("anthropic", "aws"):
                    system, anth_messages = _messages_for_anthropic(messages)
                    kwargs = {"model": model, "messages": anth_messages, "max_tokens": max_tokens}
                    if system:
                        kwargs["system"] = system
                    if tools:
                        kwargs["tools"] = _anthropic_tools_from_schema(tools)
                    if temperature is not None:
                        kwargs["temperature"] = temperature
                    if response_format:
                        schema_body = response_format.get("json_schema", {}).get("schema", {})
                        schema_name = response_format.get("json_schema", {}).get("name", "structured_output")
                        so_tool = {
                            "name": schema_name,
                            "description": "Respond with structured JSON matching this schema.",
                            "input_schema": schema_body,
                        }
                        kwargs.setdefault("tools", [])
                        kwargs["tools"].append(so_tool)
                        kwargs["tool_choice"] = {"type": "tool", "name": schema_name}
                    resp = await client.messages.create(**kwargs)
                    normalized = _normalize_anthropic_response(resp)
                    if response_format and normalized["tool_calls"]:
                        schema_name = response_format.get("json_schema", {}).get("name", "structured_output")
                        for tc in normalized["tool_calls"]:
                            if tc["function"]["name"] == schema_name:
                                normalized["content"] = tc["function"]["arguments"]
                                normalized["tool_calls"] = []
                                break
                    return normalized

                elif provider == "gcp":
                    system, contents = _messages_for_gemini(messages)
                    config = types.GenerateContentConfig(max_output_tokens=max_tokens)
                    if temperature is not None:
                        config.temperature = temperature
                    if system:
                        config.system_instruction = system
                    if tools:
                        config.tools = _gemini_tools_from_schema(tools)
                    if response_format:
                        config.response_mime_type = "application/json"
                        schema_body = response_format.get("json_schema", {}).get("schema")
                        if schema_body:
                            config.response_schema = schema_body
                    resp = await client.aio.models.generate_content(
                        model=model, contents=contents, config=config,
                    )
                    return _normalize_gemini_response(resp)

                elif provider == "aws-bedrock":
                    system, br_messages = _messages_for_bedrock(messages)
                    kwargs = {
                        "modelId": model,
                        "messages": br_messages,
                        "inferenceConfig": {"maxTokens": max_tokens},
                    }
                    if system:
                        kwargs["system"] = system
                    if temperature is not None:
                        kwargs["inferenceConfig"]["temperature"] = temperature
                    if tools:
                        kwargs["toolConfig"] = _bedrock_tools_from_schema(tools)
                    if response_format:
                        schema_body = response_format.get("json_schema", {}).get("schema", {})
                        schema_name = response_format.get("json_schema", {}).get("name", "structured_output")
                        so_spec = {"toolSpec": {
                            "name": schema_name,
                            "description": "Respond with structured JSON matching this schema.",
                            "inputSchema": {"json": schema_body},
                        }}
                        tc = kwargs.get("toolConfig", {"tools": []})
                        tc["tools"].append(so_spec)
                        tc["toolChoice"] = {"tool": {"name": schema_name}}
                        kwargs["toolConfig"] = tc
                    resp = await asyncio.to_thread(bedrock_client.converse, **kwargs)
                    normalized = _normalize_bedrock_response(resp)
                    if response_format and normalized["tool_calls"]:
                        schema_name = response_format.get("json_schema", {}).get("name", "structured_output")
                        for tc in normalized["tool_calls"]:
                            if tc["function"]["name"] == schema_name:
                                normalized["content"] = tc["function"]["arguments"]
                                normalized["tool_calls"] = []
                                break
                    return normalized

                else:
                    raise ValueError(f"Unsupported provider for tool completion: {provider}")

        except Exception as e:
            logger.info(f"fetch_tool_completion error (attempt {attempt+1}): {e}")
            wait_time = 2 ** attempt + random.uniform(0, 1)
            logger.info(f"Retrying in {wait_time:.2f}s...")
            await asyncio.sleep(wait_time)

    logger.info("fetch_tool_completion: max retries reached")
    return {"role": "assistant", "content": "Error: max retries reached", "tool_calls": []}


async def fetch_completions(client: AsyncOpenAI | AsyncTogether | genai.Client | AsyncAnthropic, prompts_lst: list[str], **kwargs) -> list[list[str]]:
    """
    e.g.
    prompts_lst = [
        "Tell me a joke.",
        "Explain quantum mechanics simply.",
        "Give me a startup idea.",
        "What's the capital of France?",
        "How do I improve memory?",
    ]

    returns responses = [
        [ # prompts_lst[0]
            "response0",
            "response1",
            ...
        ],
        [ # prompts_lst[1]
            "response0",
            "response1",
            ...
        ],
        ...
    ]
    """
    MAX_CONCURRENT_REQUESTS = 9
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Get n parameter for multiple candidates
    n = kwargs.get("n", 1)
    if isinstance(client, AsyncOpenAI) or isinstance(client, AsyncTogether) or isinstance(client, AsyncAnthropic) or isinstance(client, anthropic.AsyncAnthropicBedrock):
        processed_kwargs = {k: v for k, v in kwargs.items() if k != "n"}
    else:
        processed_kwargs = {k: v for k, v in kwargs.items()}
    
    responses = []
    # Create tasks: n tasks per prompt for multiple candidates
    tasks = []
    for prompt in prompts_lst:
        for _ in range(n):
            tasks.append(fetch_completion(semaphore, client, prompt, **processed_kwargs))
    
    results = await asyncio.gather(*tasks)
    
    # Group results by prompt (n results per prompt)
    for i, prompt in enumerate(prompts_lst):
        this_msg_choices = []
        prompt_results = results[i * n:(i + 1) * n]
        for resp in prompt_results:
            if isinstance(client, genai.Client):
                choices = [c.content.parts[0].text for c in resp.candidates]
                this_msg_choices.extend(choices)
            elif isinstance(client, AsyncTogether):
                # Together uses chat completions API
                if hasattr(resp, 'choices') and resp.choices:
                    for choice in resp.choices:
                        content = choice.message.content
                        if "</think>" in content:
                            this_msg_choices.append(content.split("</think>")[-1].strip())
                        else:
                            this_msg_choices.append(content)
                else:
                    this_msg_choices.append(str(resp))
            elif isinstance(client, AsyncAnthropic) or isinstance(client, anthropic.AsyncAnthropicBedrock):
                # Anthropic uses messages API
                if hasattr(resp, 'content') and resp.content:
                    # Extract text from content blocks
                    for block in resp.content:
                        if hasattr(block, 'text'):
                            this_msg_choices.append(block.text)
                        elif isinstance(block, dict) and 'text' in block:
                            this_msg_choices.append(block['text'])
                else:
                    this_msg_choices.append(str(resp))
            else: # OpenAI API
                # For responses API, extract from response structure
                if hasattr(resp, 'output_text'):
                    text = resp.output_text
                    this_msg_choices.append(text)
                elif hasattr(resp, 'choices'):
                    choices = [resp.choices[j].message.content for j in range(len(resp.choices))]
                    this_msg_choices.extend(choices)
                else:
                    this_msg_choices.append(str(resp))
        responses.append(this_msg_choices)
    return responses

class LLMClient():
    def __init__(self, model: str, provider: str | None = None):
        self.model = model
        self.api_model_name = model
        self.client = None
        self.async_client = None
        self._vllm_api_base = None
        # Persistent event loop so async clients can be reused across calls
        self._loop = asyncio.new_event_loop()

        self.provider = provider
        if self.provider is None:
            if "gpt" in model and "gpt-oss" not in model:
                self.provider = "openai"
            elif re.search(r"o\d", model[:2]):
                self.provider = "openai"
            elif "claude" in model:
                self.provider = "aws"
            elif "gemini" in model:
                self.provider = "gcp"

        if self.provider == "openai":
            self.client = OpenAI(api_key=openai_key_str)
            self.async_client = AsyncOpenAI(api_key=openai_key_str)
        elif self.provider == "gcp":
            self.async_client = genai.Client(vertexai=True, project=google_cloud_project, location=google_cloud_location)
        # elif self.provider == "mistralgcp":
        #     self.client = MistralGoogleCloud(region=google_cloud_region, location=google_cloud_location, project_id=google_cloud_project)
        elif self.provider == "anthropic":
            self.async_client = anthropic.AsyncAnthropic(api_key=anthropic_key_str)
        elif self.provider == "aws" and ("claude" in model or "anthropic" in model):
            self.async_client = anthropic.AsyncAnthropicBedrock(
                aws_access_key=aws_access_key,
                aws_secret_key=aws_secret_key,
                aws_region=aws_region,
            )
        elif self.provider == "aws":
            # Generic Bedrock models (Llama, Mistral, Nova, etc.) via Converse API
            self._bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name=aws_region,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
            )
            self.provider = "aws-bedrock"
        elif self.provider == "together":
            self.async_client = AsyncTogether(api_key=together_key_str)
        elif self.provider is not None and self.provider.startswith("vllm"):
            openai_api_key = "EMPTY"
            # Support per-model base URL via "vllm@<base_url>" provider format
            if "@" in self.provider:
                openai_api_base = self.provider.split("@", 1)[1]
            else:
                openai_api_base = vllm_api_base
            self.provider = "vllm"
            self._vllm_api_base = openai_api_base
            self.client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
            self.async_client = AsyncOpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
        else:
            raise ValueError(f"Invalid provider: {self.provider}")

    def _run_async(self, coro):
        """Run an async coroutine on the persistent event loop.
        Uses a single long-lived loop so async clients can reuse connections."""
        return self._loop.run_until_complete(coro)

    def _bedrock_converse(self, prompt: str, temperature=None, max_tokens=4096) -> str:
        """Call Bedrock Converse API. Works for any Bedrock model (Llama, Mistral, Nova, etc.)."""
        inference_config = {"maxTokens": max_tokens}
        if temperature is not None:
            inference_config["temperature"] = temperature
        response = self._bedrock_client.converse(
            modelId=self.model,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig=inference_config,
        )
        return response["output"]["message"]["content"][0]["text"]

    def web_search(self, query: str) -> str:
        if can_web_search_openai(self.model):
            model = self.model
            client = self.client
        else:
            model = "gpt-5.1"
            client = OpenAI(api_key=openai_key_str)
        response = client.responses.create(
            model=model,
            tools=[{ "type": "web_search_preview" }],
            input=query,
        )
        return response.output_text
    
    def web_search_async(self, queries: list[str], **kwargs) -> list[str]:
        """
        Async web search for multiple queries.
        
        Args:
            queries: List of search queries
            **kwargs: Additional parameters for the API calls
        
        Returns:
            List of web search results as strings
        
        Example:
            client = LLMClient("gpt-4o")
            queries = ["Python async best practices", "GPU optimization 2024"]
            results = await client.web_search_async(queries)
        
        Note:
            Currently supports OpenAI and Together clients only.
            Gemini support is not implemented yet.
        """
        if self.async_client is not None and isinstance(self.async_client, (AsyncOpenAI, AsyncTogether)):
            kwargs.setdefault("model", self.model)
            return self._run_async(fetch_web_search_completions(self.async_client, queries, **kwargs))
        elif isinstance(self.async_client, genai.Client):
            # Gemini not supported for web search yet
            logger.info("Web search not supported for Gemini clients yet")
            return ["Web search not supported for this client"] * len(queries)
        else:
            # Fallback to synchronous calls if no async client or unsupported client
            results = []
            for query in queries:
                try:
                    result = self.web_search(query)
                    results.append(result)
                except Exception as e:
                    logger.info(f"Web search failed for query '{query}': {e}")
                    results.append("Web search failed")
            return results
    
    def chat_async(self, prompts_lst: list[str], num_candidates=10, temperature=None, reasoning_effort="high") -> list[list[str]]:
        if self.provider == "aws-bedrock":
            # Generic Bedrock models use boto3 Converse API (synchronous),
            # wrapped with asyncio.to_thread for concurrency.
            async def _run():
                semaphore = asyncio.Semaphore(9)
                async def _call(p):
                    async with semaphore:
                        return await asyncio.to_thread(self._bedrock_converse, p, temperature)
                tasks = []
                for prompt in prompts_lst:
                    for _ in range(num_candidates):
                        tasks.append(_call(prompt))
                results = await asyncio.gather(*tasks)
                responses = []
                for i in range(len(prompts_lst)):
                    responses.append(list(results[i * num_candidates:(i + 1) * num_candidates]))
                return responses
            return self._run_async(_run())
        elif self.async_client is not None:
            # Limit concurrent requests (adjust based on your API limits)
            kwargs = {
                "model":self.model.replace("_", "/"),
                "n":num_candidates,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            if "kevin" in self.model:
                kwargs["max_tokens"] = 16384
            if is_openai_reasoning_model(self.model) and reasoning_effort is not None:
                kwargs["reasoning"] = {"effort": reasoning_effort}
            responses = self._run_async(fetch_completions(self.async_client, prompts_lst, **kwargs))
            return responses
        else:
            responses = []
            for prompt in prompts_lst:
                this_msg_resps = self.chat(prompt, num_candidates, temperature)
                responses.append(this_msg_resps)
            return responses

    def chat(self, prompt: str, num_candidates=10, temperature=None):
        responses = []
        if isinstance(self.client, Together):
            # Together uses chat completions API with messages format
            kwargs = {
                "model":self.model.replace("_", "/"),
                "messages":[{"role": "user", "content": prompt}],
                "n":num_candidates,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            if "kevin" in self.model:
                kwargs["max_tokens"] = 8192
                kwargs["timeout"] = 1200
            together_response = completions_with_backoff(self.client, **kwargs)
            for c in together_response.choices:
                responses.append(c.message.content)
        elif isinstance(self.client, OpenAI):
            kwargs = {
                "model":self.model.replace("_", "/"),
                "input":prompt,
                "n":num_candidates,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            if "kevin" in self.model:
                kwargs["max_tokens"] = 8192
                kwargs["timeout"] = 1200
            if is_openai_reasoning_model(self.model):
                kwargs["reasoning"] = {"effort": "high"}
                # Query 8 plans at a time - make multiple calls for responses API
                single_kwargs = {k: v for k, v in kwargs.items() if k != "n"}
                while len(responses) < num_candidates:
                    openai_response = completions_with_backoff(self.client, **single_kwargs)
                    if hasattr(openai_response, 'output_text'):
                        responses.append(openai_response.output_text)
                    elif hasattr(openai_response, 'choices'):
                        for c in openai_response.choices:
                            responses.append(c.message.content)
                    else:
                        responses.append(str(openai_response))
            else:
                # For responses API, make multiple calls if n > 1
                if num_candidates > 1:
                    for _ in range(num_candidates):
                        single_kwargs = {k: v for k, v in kwargs.items() if k != "n"}
                        openai_response = completions_with_backoff(self.client, **single_kwargs)
                        if hasattr(openai_response, 'output_text'):
                            responses.append(openai_response.output_text)
                        elif hasattr(openai_response, 'choices'):
                            for c in openai_response.choices:
                                responses.append(c.message.content)
                        else:
                            responses.append(str(openai_response))
                else:
                    openai_response = completions_with_backoff(self.client, **kwargs)
                    if hasattr(openai_response, 'output_text'):
                        responses.append(openai_response.output_text)
                    elif hasattr(openai_response, 'choices'):
                        for c in openai_response.choices:
                            responses.append(c.message.content)
                    else:
                        responses.append(str(openai_response))
        elif isinstance(self.client, MistralGoogleCloud):
            for _ in range(num_candidates):
                mistral_response = self.client.chat.complete(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    n=1,
                    temperature=temperature,
                )
                responses.append(mistral_response.choices[0].message.content)
        elif isinstance(self.client, genai.Client):
            # call separately since candidates are limited to 8
            gemini_response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    candidate_count=num_candidates,
                ),
            )
            for c in gemini_response.candidates:
                responses.append(c.content.parts[0].text)
        elif isinstance(self.client, anthropic.Anthropic) or isinstance(self.client, anthropic.AsyncAnthropicBedrock):
            for _ in range(num_candidates):
                claude_response = self.client.messages.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=10000,
                    # thinking={
                    #     "type": "enabled",
                    #     "budget_tokens": 2048 # relatively small budget
                    # },
                )
                responses.append(claude_response.content[-1].text)
        elif self.provider == "aws-bedrock":
            for _ in range(num_candidates):
                responses.append(self._bedrock_converse(prompt, temperature=temperature))

        return responses

    def chat_messages(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        temperature: float | None = None,
        max_tokens: int = 16384,
    ) -> dict:
        """Single LLM call with message array, optional tool schemas, optional structured output.
        Returns normalized dict: {"role": "assistant", "content": ..., "tool_calls": [...]}."""
        semaphore = asyncio.Semaphore(1)
        bedrock = getattr(self, "_bedrock_client", None)
        return self._run_async(fetch_tool_completion(
            semaphore,
            self.async_client,
            messages,
            provider=self.provider,
            model=self.model,
            tools=tools,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            bedrock_client=bedrock,
        ))

    def agent_loop(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_executor,
        max_turns: int = 30,
        temperature: float | None = None,
        max_tokens: int = 16384,
    ) -> list[dict]:
        """Run a tool-use loop: call LLM, execute tool calls, feed results back, repeat.
        Stops when the model returns content without tool calls, or max_turns is reached.
        Returns the full message history."""
        messages = copy.deepcopy(messages)
        for _ in range(max_turns):
            response = self.chat_messages(
                messages, tools=tools,
                temperature=temperature, max_tokens=max_tokens,
            )
            messages.append(response)
            if not response.get("tool_calls"):
                break
            for tc in response["tool_calls"]:
                args = tc["function"]["arguments"]
                parsed_args = json.loads(args) if isinstance(args, str) else args
                result = tool_executor(tc["function"]["name"], parsed_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": str(result),
                })
        return messages

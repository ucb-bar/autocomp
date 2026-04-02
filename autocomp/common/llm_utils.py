import os
import json
import asyncio
import random
import copy

import boto3
from openai import OpenAI, AsyncOpenAI
from google import genai
from google.genai import types
import anthropic
from anthropic import AsyncAnthropic
from together import Together, AsyncTogether

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
google_api_key = _get_key("GOOGLE_API_KEY", default=None)
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
    "GOOGLE_API_KEY": google_api_key is not None,
}
_available = [k for k, v in _key_status.items() if v]
_unavailable = [k for k, v in _key_status.items() if not v]
if _available:
    logger.info(f"Keys available: {', '.join(_available)}")
if _unavailable:
    logger.info(f"Keys unavailable: {', '.join(_unavailable)}")


def is_openai_reasoning_model(model: str) -> bool:
    return model.startswith(("o1", "o3", "o4", "gpt-5"))


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
        out.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            }
        )
    return out


def _gemini_tools_from_schema(tools: list[dict]) -> list[types.Tool]:
    """Convert OpenAI tool schema to Gemini FunctionDeclaration list."""
    declarations = []
    for t in tools:
        fn = t["function"]
        declarations.append(
            types.FunctionDeclaration(
                name=fn["name"],
                description=fn.get("description", ""),
                parameters=fn.get("parameters"),
            )
        )
    return [types.Tool(function_declarations=declarations)]


def _bedrock_tools_from_schema(tools: list[dict]) -> dict:
    """Convert OpenAI tool schema to Bedrock Converse toolConfig."""
    tool_specs = []
    for t in tools:
        fn = t["function"]
        tool_specs.append(
            {
                "toolSpec": {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "inputSchema": {
                        "json": fn.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                    },
                }
            }
        )
    return {"tools": tool_specs}


def _normalize_openai_response(message) -> dict:
    """Normalize an OpenAI/vLLM/Together ChatCompletion message to common dict."""
    tool_calls = []
    if message.tool_calls:
        for tc in message.tool_calls:
            tool_calls.append(
                {
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
            )
    return {"role": "assistant", "content": message.content, "tool_calls": tool_calls}


def _messages_for_openai_responses(
    messages: list[dict],
) -> tuple[str | None, list[dict]]:
    """Extract system→instructions; convert tool history to Responses API format."""
    instructions = None
    input_items = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            instructions = m["content"]
        elif role == "tool":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": m["tool_call_id"],
                    "output": m["content"],
                }
            )
        elif role == "assistant" and m.get("tool_calls"):
            if m.get("content"):
                input_items.append({"role": "assistant", "content": m["content"]})
            for tc in m["tool_calls"]:
                args = tc["function"]["arguments"]
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": tc["id"],
                        "name": tc["function"]["name"],
                        "arguments": args
                        if isinstance(args, str)
                        else json.dumps(args),
                    }
                )
        else:
            input_items.append({"role": role, "content": m.get("content", "")})
    return instructions, input_items


def _normalize_openai_responses_response(response) -> dict:
    """Normalize Responses API output to common dict format."""
    content_text = ""
    tool_calls = []
    for item in response.output:
        if item.type == "message":
            for part in item.content:
                if part.type == "output_text":
                    content_text += part.text
        elif item.type == "function_call":
            tool_calls.append(
                {
                    "id": item.call_id,
                    "function": {"name": item.name, "arguments": item.arguments},
                }
            )
    return {
        "role": "assistant",
        "content": content_text or None,
        "tool_calls": tool_calls,
    }


def _normalize_anthropic_response(response) -> dict:
    """Normalize an Anthropic Messages response to common dict."""
    content_text = ""
    tool_calls = []
    for block in response.content:
        if block.type == "text":
            content_text += block.text
        elif block.type == "tool_use":
            tool_calls.append(
                {
                    "id": block.id,
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                }
            )
    return {
        "role": "assistant",
        "content": content_text or None,
        "tool_calls": tool_calls,
    }


def _normalize_gemini_response(response) -> dict:
    """Normalize a Gemini generateContent response to common dict."""
    content_text = ""
    tool_calls = []
    for part in response.candidates[0].content.parts:
        if part.text:
            content_text += part.text
        if part.function_call:
            fc = part.function_call
            tool_calls.append(
                {
                    "id": f"gemini_{fc.name}_{id(fc)}",
                    "function": {
                        "name": fc.name,
                        "arguments": json.dumps(dict(fc.args) if fc.args else {}),
                    },
                }
            )
    return {
        "role": "assistant",
        "content": content_text or None,
        "tool_calls": tool_calls,
    }


def _normalize_bedrock_response(response) -> dict:
    """Normalize a Bedrock Converse response to common dict."""
    content_text = ""
    tool_calls = []
    for block in response["output"]["message"]["content"]:
        if "text" in block:
            content_text += block["text"]
        elif "toolUse" in block:
            tu = block["toolUse"]
            tool_calls.append(
                {
                    "id": tu["toolUseId"],
                    "function": {
                        "name": tu["name"],
                        "arguments": json.dumps(tu["input"]),
                    },
                }
            )
    return {
        "role": "assistant",
        "content": content_text or None,
        "tool_calls": tool_calls,
    }


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
            if (
                out
                and out[-1]["role"] == "user"
                and isinstance(out[-1]["content"], list)
                and out[-1]["content"]
                and out[-1]["content"][0].get("type") == "tool_result"
            ):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif m["role"] == "assistant" and m.get("tool_calls"):
            content = []
            if m.get("content"):
                content.append({"type": "text", "text": m["content"]})
            for tc in m["tool_calls"]:
                args = tc["function"]["arguments"]
                content.append(
                    {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": json.loads(args) if isinstance(args, str) else args,
                    }
                )
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
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=fn_name,
                                response={"result": m["content"]},
                            )
                        )
                    ],
                )
            )
        elif m["role"] == "assistant" and m.get("tool_calls"):
            parts = []
            if m.get("content"):
                parts.append(types.Part(text=m["content"]))
            for tc in m["tool_calls"]:
                args = tc["function"]["arguments"]
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(
                            name=tc["function"]["name"],
                            args=json.loads(args) if isinstance(args, str) else args,
                        )
                    )
                )
            contents.append(types.Content(role="model", parts=parts))
        else:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(
                types.Content(role=role, parts=[types.Part(text=m["content"])])
            )
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
            block = {
                "toolResult": {
                    "toolUseId": m["tool_call_id"],
                    "content": [{"text": m["content"]}],
                }
            }
            if (
                out
                and out[-1]["role"] == "user"
                and isinstance(out[-1]["content"], list)
                and out[-1]["content"]
                and "toolResult" in out[-1]["content"][0]
            ):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif m["role"] == "assistant" and m.get("tool_calls"):
            content = []
            if m.get("content"):
                content.append({"text": m["content"]})
            for tc in m["tool_calls"]:
                args = tc["function"]["arguments"]
                content.append(
                    {
                        "toolUse": {
                            "toolUseId": tc["id"],
                            "name": tc["function"]["name"],
                            "input": json.loads(args)
                            if isinstance(args, str)
                            else args,
                        }
                    }
                )
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
    max_tokens: int | None = None,
    reasoning: dict | None = None,
    bedrock_client=None,
) -> dict:
    """Provider-dispatching async helper for chat-completions with tool calling.

    Returns a normalized dict: {"role": "assistant", "content": ..., "tool_calls": [...]}.
    """
    max_retries = 8
    for attempt in range(max_retries):
        try:
            async with semaphore:
                if provider == "openai":
                    instructions, input_items = _messages_for_openai_responses(messages)
                    kwargs = {"model": model, "input": input_items}
                    if instructions:
                        kwargs["instructions"] = instructions
                    if max_tokens is not None:
                        kwargs["max_tokens"] = max_tokens
                    if tools:

                        def _conv(t):
                            if "function" not in t:
                                return t
                            fn = t["function"]
                            return {
                                "type": "function",
                                "name": fn["name"],
                                "description": fn.get("description", ""),
                                "parameters": fn.get("parameters", {}),
                                "strict": fn.get("strict", False),
                            }

                        kwargs["tools"] = [_conv(t) for t in tools]
                    if temperature is not None:
                        kwargs["temperature"] = temperature
                    if reasoning:
                        kwargs["reasoning"] = reasoning
                    if response_format:
                        inner = response_format.get("json_schema", {})
                        kwargs["text"] = {
                            "format": {
                                "type": "json_schema",
                                "name": inner.get("name", "structured_output"),
                                "strict": inner.get("strict", True),
                                "schema": inner.get("schema", {}),
                            }
                        }
                    resp = await client.responses.create(**kwargs)
                    return _normalize_openai_responses_response(resp)

                elif provider in ("vllm", "together"):
                    kwargs = {"model": model, "messages": messages}
                    if max_tokens is not None:
                        kwargs["max_tokens"] = max_tokens
                    if tools:
                        kwargs["tools"] = _openai_tools_from_schema(tools)
                    if temperature is not None:
                        kwargs["temperature"] = temperature
                    if response_format:
                        kwargs["response_format"] = response_format
                    resp = await client.chat.completions.create(**kwargs)
                    return _normalize_openai_response(resp.choices[0].message)

                elif provider in ("anthropic", "aws"):
                    system, anth_messages = _messages_for_anthropic(messages)
                    kwargs = {
                        "model": model,
                        "messages": anth_messages,
                        "max_tokens": max_tokens or 16384,
                    }
                    if system:
                        kwargs["system"] = system
                    if tools:
                        kwargs["tools"] = _anthropic_tools_from_schema(tools)
                    if temperature is not None:
                        kwargs["temperature"] = temperature
                    if response_format:
                        schema_body = response_format.get("json_schema", {}).get(
                            "schema", {}
                        )
                        schema_name = response_format.get("json_schema", {}).get(
                            "name", "structured_output"
                        )
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
                        schema_name = response_format.get("json_schema", {}).get(
                            "name", "structured_output"
                        )
                        for tc in normalized["tool_calls"]:
                            if tc["function"]["name"] == schema_name:
                                normalized["content"] = tc["function"]["arguments"]
                                normalized["tool_calls"] = []
                                break
                    return normalized

                elif provider == "gcp":
                    system, contents = _messages_for_gemini(messages)
                    config = types.GenerateContentConfig()
                    if max_tokens is not None:
                        config.max_output_tokens = max_tokens
                    if temperature is not None:
                        config.temperature = temperature
                    if system:
                        config.system_instruction = system
                    if tools:
                        config.tools = _gemini_tools_from_schema(tools)
                    if response_format:
                        config.response_mime_type = "application/json"
                        schema_body = response_format.get("json_schema", {}).get(
                            "schema"
                        )
                        if schema_body:
                            config.response_schema = schema_body
                    resp = await client.aio.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config,
                    )
                    return _normalize_gemini_response(resp)

                elif provider == "aws-bedrock":
                    system, br_messages = _messages_for_bedrock(messages)
                    kwargs = {
                        "modelId": model,
                        "messages": br_messages,
                        "inferenceConfig": {},
                    }
                    if max_tokens is not None:
                        kwargs["inferenceConfig"]["maxTokens"] = max_tokens
                    if system:
                        kwargs["system"] = system
                    if temperature is not None:
                        kwargs["inferenceConfig"]["temperature"] = temperature
                    if tools:
                        kwargs["toolConfig"] = _bedrock_tools_from_schema(tools)
                    if response_format:
                        schema_body = response_format.get("json_schema", {}).get(
                            "schema", {}
                        )
                        schema_name = response_format.get("json_schema", {}).get(
                            "name", "structured_output"
                        )
                        so_spec = {
                            "toolSpec": {
                                "name": schema_name,
                                "description": "Respond with structured JSON matching this schema.",
                                "inputSchema": {"json": schema_body},
                            }
                        }
                        tc = kwargs.get("toolConfig", {"tools": []})
                        tc["tools"].append(so_spec)
                        tc["toolChoice"] = {"tool": {"name": schema_name}}
                        kwargs["toolConfig"] = tc
                    resp = await asyncio.to_thread(bedrock_client.converse, **kwargs)
                    normalized = _normalize_bedrock_response(resp)
                    if response_format and normalized["tool_calls"]:
                        schema_name = response_format.get("json_schema", {}).get(
                            "name", "structured_output"
                        )
                        for tc in normalized["tool_calls"]:
                            if tc["function"]["name"] == schema_name:
                                normalized["content"] = tc["function"]["arguments"]
                                normalized["tool_calls"] = []
                                break
                    return normalized

                else:
                    raise ValueError(
                        f"Unsupported provider for tool completion: {provider} for model: {model}"
                    )

        except Exception as e:
            err_str = str(e)
            if "temperature" in err_str and "not supported" in err_str:
                logger.info(
                    f"fetch_tool_completion: model {model} does not support temperature, retrying without it"
                )
                temperature = None
                continue
            logger.info(f"fetch_tool_completion error (attempt {attempt + 1}): {e}")
            wait_time = 2**attempt + random.uniform(0, 1)
            logger.info(f"Retrying in {wait_time:.2f}s...")
            await asyncio.sleep(wait_time)

    logger.info("fetch_tool_completion: max retries reached")
    return {
        "role": "assistant",
        "content": "Error: max retries reached",
        "tool_calls": [],
    }


class LLMClient:
    def __init__(self, model: str, provider: str | None = None):
        self.model = model
        self.client = None
        self.async_client = None
        self._vllm_api_base = None
        # Persistent event loop so async clients can be reused across calls
        self._loop = asyncio.new_event_loop()

        self.provider = provider
        if self.provider is None:
            if "gpt" in model and "gpt-oss" not in model:
                self.provider = "openai"
            elif len(model) >= 2 and model[0] == "o" and model[1].isdigit():
                self.provider = "openai"
            elif "claude" in model:
                self.provider = "aws"
            elif "gemini" in model:
                self.provider = "gcp"

        if self.provider == "openai":
            self.client = OpenAI(api_key=openai_key_str)
            self.async_client = AsyncOpenAI(api_key=openai_key_str)
        elif self.provider == "gcp":
            if google_api_key and not google_cloud_project:
                self.client = genai.Client(api_key=google_api_key)
            else:
                self.client = genai.Client(
                    vertexai=True,
                    project=google_cloud_project,
                    location=google_cloud_location,
                )
            self.async_client = self.client
        elif self.provider == "anthropic":
            self.async_client = anthropic.AsyncAnthropic(api_key=anthropic_key_str)
        elif self.provider == "aws" and ("claude" in model or "anthropic" in model):
            # Use explicit keys if provided, otherwise let boto3/anthropic
            # pick up credentials from IAM role (instance metadata)
            bedrock_kwargs = {"aws_region": aws_region}
            if aws_access_key and aws_secret_key:
                bedrock_kwargs["aws_access_key"] = aws_access_key
                bedrock_kwargs["aws_secret_key"] = aws_secret_key
            self.client = anthropic.AnthropicBedrock(**bedrock_kwargs)
            self.async_client = anthropic.AsyncAnthropicBedrock(**bedrock_kwargs)
        elif self.provider == "aws":
            # Generic Bedrock models (Llama, Mistral, Nova, etc.) via Converse API
            # Use explicit keys if provided, otherwise IAM role
            from botocore.config import Config as BotoConfig

            boto_kwargs = {
                "region_name": aws_region,
                "config": BotoConfig(read_timeout=120),
            }
            if aws_access_key and aws_secret_key:
                boto_kwargs["aws_access_key_id"] = aws_access_key
                boto_kwargs["aws_secret_access_key"] = aws_secret_key
            self._bedrock_client = boto3.client("bedrock-runtime", **boto_kwargs)
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
        elif self.provider == "dummy":
            pass
        else:
            raise ValueError(
                f"Invalid provider: {self.provider} for model: {self.model}"
            )

    def _run_async(self, coro):
        """Run an async coroutine on the persistent event loop.
        Uses a single long-lived loop so async clients can reuse connections."""
        return self._loop.run_until_complete(coro)

    def web_search(self, query: str) -> str:
        messages = [{"role": "user", "content": query}]
        tools = [{"type": "web_search_preview"}]
        result = self.chat_messages(messages, tools=tools)
        return result.get("content") or ""

    def web_search_async(self, queries: list[str], **kwargs) -> list[str]:
        messages_lst = [[{"role": "user", "content": q}] for q in queries]
        tools = [{"type": "web_search_preview"}]
        grouped = self.chat_messages_async(messages_lst, num_samples=1, tools=tools)
        return [r[0].get("content") or "" for r in grouped]

    def chat_async(
        self,
        prompts_lst: list[str],
        num_samples=10,
        temperature=None,
        reasoning_effort="high",
    ) -> list[list[str]]:
        if self.provider == "dummy":
            return [["dummy response"] * num_samples for _ in prompts_lst]
        messages_lst = [[{"role": "user", "content": p}] for p in prompts_lst]
        reasoning = None
        if is_openai_reasoning_model(self.model) and reasoning_effort is not None:
            reasoning = {"effort": reasoning_effort}
            temperature = None
        grouped = self.chat_messages_async(
            messages_lst,
            num_samples=num_samples,
            temperature=temperature,
            reasoning=reasoning,
        )
        return [[r.get("content") or "" for r in samples] for samples in grouped]

    def chat(self, prompt: str, num_samples=10, temperature=None):
        """Synchronous convenience wrapper. Returns list of response strings."""
        return self.chat_async(
            [prompt], num_samples=num_samples, temperature=temperature
        )[0]

    def chat_messages(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning: dict | None = None,
    ) -> dict:
        """Single LLM call with message array, optional tool schemas, optional structured output.
        Returns normalized dict: {"role": "assistant", "content": ..., "tool_calls": [...]}."""
        semaphore = asyncio.Semaphore(1)
        bedrock = getattr(self, "_bedrock_client", None)
        return self._run_async(
            fetch_tool_completion(
                semaphore,
                self.async_client,
                messages,
                provider=self.provider,
                model=self.model,
                tools=tools,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning=reasoning,
                bedrock_client=bedrock,
            )
        )

    def chat_messages_async(
        self,
        messages_lst: list[list[dict]],
        num_samples: int = 1,
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning: dict | None = None,
    ) -> list[list[dict]]:
        """Batched async version of chat_messages.

        Args:
            messages_lst: List of message arrays, one per prompt.
            num_samples: Number of samples per prompt.
            tools: Optional tool schemas.
            response_format: Optional structured output schema.
            temperature: Sampling temperature.
            max_tokens: Max tokens per response.

        Returns:
            List of lists of normalized dicts, one inner list per prompt,
            each inner list containing num_samples responses.
        """
        MAX_CONCURRENT = 9
        bedrock = getattr(self, "_bedrock_client", None)

        async def _run():
            semaphore = asyncio.Semaphore(MAX_CONCURRENT)
            tasks = []
            for messages in messages_lst:
                for _ in range(num_samples):
                    tasks.append(
                        fetch_tool_completion(
                            semaphore,
                            self.async_client,
                            copy.deepcopy(messages),
                            provider=self.provider,
                            model=self.model,
                            tools=tools,
                            response_format=response_format,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            reasoning=reasoning,
                            bedrock_client=bedrock,
                        )
                    )
            return await asyncio.gather(*tasks)

        results = self._run_async(_run())

        grouped: list[list[dict]] = []
        for i in range(len(messages_lst)):
            grouped.append(list(results[i * num_samples : (i + 1) * num_samples]))
        return grouped

    def agent_loop(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_executor,
        max_turns: int = 30,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> list[dict]:
        """Run a tool-use loop: call LLM, execute tool calls, feed results back, repeat.
        Stops when the model returns content without tool calls, or max_turns is reached.
        Returns the full message history."""
        messages = copy.deepcopy(messages)
        for _ in range(max_turns):
            response = self.chat_messages(
                messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            messages.append(response)
            if not response.get("tool_calls"):
                break
            for tc in response["tool_calls"]:
                args = tc["function"]["arguments"]
                parsed_args = json.loads(args) if isinstance(args, str) else args
                result = tool_executor(tc["function"]["name"], parsed_args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": str(result),
                    }
                )
        return messages

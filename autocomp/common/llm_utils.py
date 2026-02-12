import os
import re
import asyncio
import random

import backoff
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

        self.provider = provider
        if self.provider is None:
            # Try to guess the provider based on the model name
            if "gpt" in model and "gpt-oss" not in model:
                self.provider = "openai"
            elif re.search(r"o\d", model[:2]):
                self.provider = "openai"
            elif "claude" in model:
                self.provider = "aws"
            elif "gemini" in model:
                self.provider = "gcp"
        elif self.provider == "openai":
            self.client = OpenAI(api_key=openai_key_str)
            self.async_client = AsyncOpenAI(api_key=openai_key_str)
        elif self.provider == "gcp":
            # genai.configure(api_key=gemini_key_str)
            # self.client = genai.GenerativeModel(model_name=model)
            self.async_client = genai.Client(vertexai=True, project=google_cloud_project, location=google_cloud_location)
        # elif self.provider == "mistralgcp":
        #     self.client = MistralGoogleCloud(region=google_cloud_region, location=google_cloud_location, project_id=google_cloud_project)
        elif self.provider == "anthropic":
            self.async_client = anthropic.AsyncAnthropic(api_key=anthropic_key_str)
        elif self.provider == "aws" and "claude" in model:
                self.async_client = anthropic.AsyncAnthropicBedrock(
                    aws_access_key=aws_access_key,
                    aws_secret_key=aws_secret_key,
                    aws_region="us-west-2",
                )
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
            return asyncio.run(fetch_web_search_completions(self.async_client, queries, **kwargs))
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
        if self.async_client is not None:
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
            responses = asyncio.run(fetch_completions(self.async_client, prompts_lst, **kwargs))
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

        return responses

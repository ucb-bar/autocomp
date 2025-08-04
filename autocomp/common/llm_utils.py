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
from together import Together, AsyncTogether
from mistralai_gcp import MistralGoogleCloud

from autocomp.common import logger

# Try environment variable first, then fallback to import
openai_key_str = os.environ.get("OPENAI_API_KEY")
if not openai_key_str:
    try:
        import autocomp.common.openai_key as openai_key
        openai_key_str = openai_key.key
    except ImportError:
        logger.info("No OpenAI key found in env or import. Continuing with empty key.")
        openai_key_str = "EMPTY"

anthropic_key_str = os.environ.get("ANTHROPIC_API_KEY")
if not anthropic_key_str:
    try:
        import autocomp.common.anthropic_key as anthropic_key
        anthropic_key_str = anthropic_key.key
    except ImportError:
        logger.info("No Anthropic key found in env or import. Continuing with empty key.")
        anthropic_key_str = "EMPTY"

gemini_key_str = os.environ.get("GOOGLE_API_KEY")
if not gemini_key_str:
    try:
        import autocomp.common.gemini_key as gemini_key
        gemini_key_str = gemini_key.key
    except ImportError:
        logger.info("No Gemini key found in env or import. Continuing with empty key.")
        gemini_key_str = "EMPTY"

together_key_str = os.environ.get("TOGETHER_API_KEY")
if not together_key_str:
    try:
        import autocomp.common.together_key as together_key
        together_key_str = together_key.key
    except ImportError:
        logger.info("No Together key found in env or import. Continuing with empty key.")
        together_key_str = "EMPTY"

google_cloud_region = os.environ.get("GOOGLE_CLOUD_REGION")
google_cloud_location = os.environ.get("GOOGLE_CLOUD_LOCATION")
google_cloud_project_id = os.environ.get("GOOGLE_CLOUD_PROJECT_ID")
if google_cloud_region is None:
    google_cloud_region = "us-central1" # your region here
if google_cloud_location is None:
    google_cloud_location = "global" # your location here
if google_cloud_project_id is None:
    google_cloud_project_id = None # your project ID here


def extract(s):
    # return [x for x in re.findall(r"```(?:python|Python)?(.*)```", s, re.DOTALL)]
    return [x for x in re.findall(r"```(?:c|c\+\+|cpp)?\n(void test\(.*}\n.*)```", s, re.DOTALL)]

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def completions_with_backoff(client: OpenAI, **kwargs):
    return client.chat.completions.create(**kwargs)

async def fetch_completion(semaphore: asyncio.Semaphore, client: OpenAI, messages, **kwargs):
    """Fetches a chat completion with retries and rate limit handling."""
    max_retries = 6
    for attempt in range(max_retries):
        try:
            async with semaphore:  # Limits concurrent requests
                response = await client.chat.completions.create(messages=messages, **kwargs)
            return response
        
        # except (RateLimitError, APITimeoutError, InternalServerError):
        except Exception as e:
            logger.info(f"Error: {e}")
            wait_time = 2 ** attempt + random.uniform(0, 1)  # Exponential backoff
            logger.info(f"Rate limit hit! Retrying in {wait_time:.2f} seconds...")
            await asyncio.sleep(wait_time)
    
    print("Max retries reached, request failed.")
    return None

async def fetch_completions(client: OpenAI, msgs_lst: list[list[dict]], **kwargs) -> list[list[str]]:
    """
    e.g.
    msgs_lst = [
        [{"role": "user", "content": "Tell me a joke."}],
        [{"role": "user", "content": "Explain quantum mechanics simply."}],
        [{"role": "user", "content": "Give me a startup idea."}],
        [{"role": "user", "content": "What's the capital of France?"}],
        [{"role": "user", "content": "How do I improve memory?"}],
    ]
    """
    MAX_CONCURRENT_REQUESTS = 8
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    responses = []
    tasks = [fetch_completion(semaphore, client, messages, **kwargs) for messages in msgs_lst]
    results = await asyncio.gather(*tasks)
    for resp in results:
        this_msg_choices = []
        for c in resp.choices:
            this_msg_choices.append(c.message.content)
        responses.append(this_msg_choices)
    return responses

class LLMClient():
    def __init__(self, model: str):
        self.model = model
        self.async_client = None
        if "gpt" in model or re.search(r"o\d", model[:2]):
            self.client = OpenAI(api_key=openai_key_str)
            self.async_client = AsyncOpenAI(api_key=openai_key_str)
        elif "gemini" in model:
            # genai.configure(api_key=gemini_key_str)
            # self.client = genai.GenerativeModel(model_name=model)
            self.client = genai.Client(vertexai=True, project=google_cloud_project_id, location=google_cloud_region)
        elif "claude" in model:
            self.client = anthropic.Anthropic(api_key=anthropic_key_str)
        elif "Qwen" in model or "llama" in model:
            self.async_client = AsyncTogether(api_key=together_key_str)
        elif "mistral" in model or "mixtral" in model:
            self.client = MistralGoogleCloud(region=google_cloud_region, project_id=google_cloud_project_id)
        elif "gemma" in model:
            openai_api_key = "EMPTY"
            openai_api_base = "http://localhost:8000/v1"
            self.client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
            model = self.client.models.list()
            self.model = model.data[0].id

    def chat_async(self, msgs_lst: list[list[dict]], num_candidates=10, temperature=0.7) -> list[list[str]]:
        if self.async_client is not None:
            # Limit concurrent requests (adjust based on your API limits)
            kwargs = {
                "model":self.model.replace("_", "/"),
                "n":num_candidates,
                "temperature":temperature,
            }
            if "o1" in self.model or "o3" in self.model:
                kwargs["reasoning_effort"] = "high"
            responses = asyncio.run(fetch_completions(self.async_client, msgs_lst, **kwargs))
            return responses
        else:
            responses = []
            for messages in msgs_lst:
                this_msg_resps = self.chat(messages, num_candidates, temperature)
                responses.append(this_msg_resps)
            return responses

    def chat(self, messages: list, num_candidates=10, temperature=0.7):
        responses = []
        if isinstance(self.client, OpenAI) or isinstance(self.client, Together):
            kwargs = {
                "model":self.model.replace("_", "/"),
                "messages":messages,
                "n":num_candidates,
                "temperature":temperature,
            }
            if "o1" in self.model or "o3" in self.model:
                kwargs["reasoning_effort"] = "high"
                # Query 8 plans at a time
                while len(responses) < num_candidates:
                    kwargs["n"] = min(8, num_candidates - len(responses))
                    openai_response = completions_with_backoff(self.client, **kwargs)
                    for c in openai_response.choices:
                        responses.append(c.message.content)
            else:
                openai_response = completions_with_backoff(self.client, **kwargs)
                for c in openai_response.choices:
                    responses.append(c.message.content)
        elif isinstance(self.client, MistralGoogleCloud):
            for _ in range(num_candidates):
                mistral_response = self.client.chat.complete(
                    model=self.model,
                    messages=messages,
                    n=1,
                    temperature=temperature,
                )
                responses.append(mistral_response.choices[0].message.content)
        elif isinstance(self.client, genai.Client):
            # call separately since candidates are limited to 8
            gemini_response = self.client.models.generate_content(
                model=self.model,
                contents="\n".join([dic["content"] for dic in messages]),
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    candidate_count=num_candidates,
                ),
            )
            for c in gemini_response.candidates:
                responses.append(c.content.parts[0].text)
        elif isinstance(self.client, anthropic.Anthropic):
            for _ in range(num_candidates):
                claude_response = self.client.messages.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=10000,
                    # thinking={
                    #     "type": "enabled",
                    #     "budget_tokens": 2048 # relatively small budget
                    # },
                )
                responses.append(claude_response.content[-1].text)

        return responses

import os
import re
import asyncio
import random
import time

import backoff
import openai
from openai import OpenAI, AsyncOpenAI, RateLimitError, APITimeoutError, InternalServerError
from google import genai
from google.genai import types
import anthropic
from together import Together, AsyncTogether
from mistralai_gcp import MistralGoogleCloud

from autocomp.common import logger
from autocomp.common.llm_queue import LLMRequestWriter

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
if not google_cloud_region:
    google_cloud_region = "us-central1" # your region here
if not google_cloud_location:
    google_cloud_location = "global" # your location here
if not google_cloud_project_id:
    google_cloud_project_id = "" # your project ID here


def extract(s):
    # return [x for x in re.findall(r"```(?:python|Python)?(.*)```", s, re.DOTALL)]
    return [x for x in re.findall(r"```(?:c|c\+\+|cpp)?\n(void test\(.*}\n.*)```", s, re.DOTALL)]

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def completions_with_backoff(client: OpenAI, **kwargs):
    return client.chat.completions.create(**kwargs)

async def fetch_completion(semaphore: asyncio.Semaphore, client: AsyncOpenAI | AsyncTogether | genai.Client, messages, **kwargs):
    """Fetches a chat completion with retries and rate limit handling."""
    max_retries = 8
    for attempt in range(max_retries):
        try:
            async with semaphore:  # Limits concurrent requests
                if isinstance(client, AsyncOpenAI) or isinstance(client, AsyncTogether):
                    response = await client.chat.completions.create(messages=messages, **kwargs)
                elif isinstance(client, genai.Client):
                    response = await client.aio.models.generate_content(
                        model=kwargs["model"],
                        contents="\n".join([dic["content"] for dic in messages]),
                        config=types.GenerateContentConfig(
                            temperature=kwargs["temperature"],
                            candidate_count=kwargs["n"],
                        ),
                    )
            return response
        
        # except (RateLimitError, APITimeoutError, InternalServerError):
        except Exception as e:
            logger.info(f"Error: {e}")
            wait_time = 2 ** attempt + random.uniform(0, 1)  # Exponential backoff
            logger.info(f"Rate limit hit! Retrying in {wait_time:.2f} seconds...")
            await asyncio.sleep(wait_time)
    
    print("Max retries reached, request failed.")
    return None

async def fetch_completions(client: AsyncOpenAI | AsyncTogether | genai.Client, msgs_lst: list[list[dict]], **kwargs) -> list[list[str]]:
    """
    e.g.
    msgs_lst = [
        [{"role": "user", "content": "Tell me a joke."}],
        [{"role": "user", "content": "Explain quantum mechanics simply."}],
        [{"role": "user", "content": "Give me a startup idea."}],
        [{"role": "user", "content": "What's the capital of France?"}],
        [{"role": "user", "content": "How do I improve memory?"}],
    ]

    returns responses = [
        [ # msgs_lst[0]
            "response0",
            "response1",
            ...
        ],
        [ # msgs_lst[1]
            "response0",
            "response1",
            ...
        ],
        ...
    ]
    """
    MAX_CONCURRENT_REQUESTS = 9
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    responses = []
    tasks = [fetch_completion(semaphore, client, messages, **kwargs) for messages in msgs_lst]
    results = await asyncio.gather(*tasks)
    for resp in results:
        this_msg_choices = []
        if isinstance(client, genai.Client):
            choices = [c.content.parts[0].text for c in resp.candidates]
        else: # OpenAI API
            choices = [resp.choices[i].message.content for i in range(len(resp.choices))]

        for c in choices:
            if "</think>" in c:
                this_msg_choices.append(c.split("</think>")[-1].strip())
            else:
                this_msg_choices.append(c)
        responses.append(this_msg_choices)
    return responses

class LLMClient():
    def __init__(self, model: str, use_queue: bool = False, queue_dir: str = None):
        self.model = model
        self.async_client = None
        self.use_queue = use_queue
        self.queue_dir = queue_dir
        if not use_queue:
            if "Qwen" in model or "llama" in model or "deepseek" in model or "gpt-oss" in model:
                self.async_client = AsyncTogether(api_key=together_key_str)
            elif "gpt" in model or re.search(r"o\d", model[:2]):
                self.client = OpenAI(api_key=openai_key_str)
                self.async_client = AsyncOpenAI(api_key=openai_key_str)
            elif "gemini" in model:
                # genai.configure(api_key=gemini_key_str)
                # self.client = genai.GenerativeModel(model_name=model)
                self.async_client = genai.Client(vertexai=True, project=google_cloud_project_id, location=google_cloud_region)
            elif "claude" in model:
                self.client = anthropic.Anthropic(api_key=anthropic_key_str)
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
        if self.use_queue:
            # Initialize queue writer with default queue directory
            writer = LLMRequestWriter(self.queue_dir)
            
            # Submit requests for each message list and each candidate
            all_request_ids = []
            for messages in msgs_lst:
                msg_request_ids = []
                for _ in range(num_candidates):
                    request_id = writer.submit_request(
                        messages=messages,
                        model=self.model.replace("_", "/"),
                        temperature=temperature,
                        max_tokens=2048,
                    )
                    msg_request_ids.append(request_id)
                all_request_ids.append(msg_request_ids)
            
            # Poll for responses
            responses = []
            max_wait_time = 300  # 5 minutes timeout
            poll_interval = 0.5  # Poll every 500ms
            
            for msg_request_ids in all_request_ids:
                msg_responses = []
                start_time = time.time()
                
                while len(msg_responses) < len(msg_request_ids) and (time.time() - start_time) < max_wait_time:
                    for request_id in msg_request_ids:
                        if request_id not in [r[1] for r in msg_responses]:  # Not already collected
                            status = writer.get_request_status(request_id)
                            if status == "completed":
                                response_obj = writer.get_response(request_id)
                                if response_obj and not response_obj.error:
                                    content = response_obj.response
                                    if "</think>" in content:
                                        content = content.split("</think>")[-1].strip()
                                    msg_responses.append((content, request_id))
                                elif response_obj and response_obj.error:
                                    logger.error(f"Request {request_id} failed: {response_obj.error}")
                                    msg_responses.append(("", request_id))  # Empty response for failed requests
                            elif status == "failed":
                                logger.error(f"Request {request_id} failed")
                                msg_responses.append(("", request_id))  # Empty response for failed requests
                    
                    if len(msg_responses) < len(msg_request_ids):
                        time.sleep(poll_interval)
                
                # Extract just the response content, maintaining order
                ordered_responses = [""] * len(msg_request_ids)
                for content, request_id in msg_responses:
                    try:
                        idx = msg_request_ids.index(request_id)
                        ordered_responses[idx] = content
                    except ValueError:
                        pass  # Request ID not found, skip
                
                responses.append(ordered_responses)
            
            return responses
        elif self.async_client is not None:
            # Limit concurrent requests (adjust based on your API limits)
            kwargs = {
                "model":self.model.replace("_", "/"),
                "n":num_candidates,
                "temperature":temperature,
            }
            if "kevin" in self.model:
                kwargs["max_tokens"] = 16384
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
            if "kevin" in self.model:
                kwargs["max_tokens"] = 8192
                kwargs["timeout"] = 1200
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

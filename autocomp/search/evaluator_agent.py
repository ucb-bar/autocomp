import re
import pathlib

from autocomp.common import LLMClient, logger
from autocomp.search.code_repo import CodeCandidate
from prompts import isa_prompt_conv

class EvaluatorAgent:
    def __init__(self, model, use_queue: bool = False, queue_dir: str = None):
        logger.debug(f"Initializing EvaluatorAgent with model {model} and use_queue {use_queue}")
        self.model = model
        self.use_queue = use_queue
        self.queue_dir = queue_dir
        self.llm_client = LLMClient(model, use_queue, queue_dir)

    def __repr__(self):
        return f"EvaluatorAgent({self.model}, {self.use_queue}, {self.queue_dir})"

    def _extract_score_robust(self, response: str) -> float:
        """
        Robustly extract a prediction from LLM response using multiple fallback methods.
        Returns a float (any numeric value), or 0.0 if parsing fails completely.
        """
        if not response or not isinstance(response, str):
            logger.warning("Empty or invalid response received")
            return 0.0
        
        response_clean = response.strip().replace("*", "")
        
        # Method 1: Look for "Prediction:" followed by a number (most common format)
        score_patterns = [
            r"Prediction:\s*(-?\d+(?:\.\d+)?)",
            r"Prediction\s*:\s*(-?\d+(?:\.\d+)?)",
            r"prediction:\s*(-?\d+(?:\.\d+)?)",
            r"prediction\s*:\s*(-?\d+(?:\.\d+)?)",
            r"Prediction\*?\*?:\s*\*?\*?(-?\d+(?:\.\d+)?)",
        ]
        
        for pattern in score_patterns:
            matches = re.findall(pattern, response_clean, re.IGNORECASE)
            if matches:
                try:
                    score = float(matches[-1])  # Take the last match
                    return score
                except ValueError:
                    continue
        
        # Method 2: Look for standalone numbers that could be scores
        # Find all numbers in the response
        number_patterns = [
            r"\b(-?\d+(?:\.\d+)?)\b",  # Any digit numbers including negative (scores can be any value)
        ]
        
        for pattern in number_patterns:
            numbers = re.findall(pattern, response_clean)
            for num_str in reversed(numbers):  # Check from end backwards
                try:
                    num = float(num_str)
                    logger.info(f"Extracted prediction {num} using fallback number pattern")
                    return num
                except ValueError:
                    continue
        
        # Method 3: Look for common score expressions
        expression_patterns = [
            r"(-?\d+(?:\.\d+)?)\s*(?:out of|/)\s*100",
            r"(-?\d+(?:\.\d+)?)\s*points?",
            r"(-?\d+(?:\.\d+)?)\s*%",
            r"rating.*?(-?\d+(?:\.\d+)?)",
            r"evaluate.*?(-?\d+(?:\.\d+)?)",
        ]
        
        for pattern in expression_patterns:
            matches = re.findall(pattern, response_clean, re.IGNORECASE)
            if matches:
                try:
                    score = float(matches[-1])
                    logger.info(f"Extracted prediction {score} using expression pattern")
                    return score
                except ValueError:
                    continue
        
        # Method 4: Try to find the last number in the response that could be a score
        all_numbers = re.findall(r"-?\d+(?:\.\d+)?", response_clean)
        if all_numbers:
            for num_str in reversed(all_numbers):
                try:
                    num = float(num_str)
                    logger.warning(f"Using last valid number {num} as fallback prediction")
                    return num
                except ValueError:
                    continue
        
        # Method 5: Parse split method (original approach) as final fallback
        try:
            if "Prediction:" in response_clean:
                score_part = response_clean.split("Prediction:")[-1]
                # Try different ways to extract the number
                for line in score_part.split("\n"):
                    line = line.strip()
                    if line:
                        # Extract first number from the line
                        numbers = re.findall(r"-?\d+(?:\.\d+)?", line)
                        if numbers:
                            score = float(numbers[0])
                            logger.warning(f"Using original split method, extracted {score}")
                            return score
        except (ValueError, IndexError):
            pass
        
        # If all methods fail, log the response and return 0
        logger.error(f"Failed to extract prediction from response: {response_clean[:200]}...")
        return 0.0

    def evaluate_plans(self, orig_codes: list[str], orig_code_latencies: list[float], plans: list[str], save_dir: pathlib.Path, save_str: str, feedbacks: list[str] = None) -> list[float]:
        messages = []
        if not feedbacks:
            feedbacks = [None] * len(orig_codes)
        for i, (orig_code, orig_code_latency, feedback, plan) in enumerate(zip(orig_codes, orig_code_latencies, feedbacks, plans)):
            prompt = isa_prompt_conv.PROMPT(16) + "\n"
            prompt += f"""You are an evaluator for a tensor processing optimization plan.
You will be given a code implementation of a tensor processing operation and a plan to optimize it.
Your task is to evaluate the plan and predict the performance improvement or degradation in percentage it will achieve.
Predict 0 if the plan will result in incorrect code.

Original code:
{orig_code}
Original code latency: {orig_code_latency} cycles
"""
            if feedback:
                prompt += f"""
Feedback:
{feedback}
"""
            prompt += f"""
Plan:
{plan}

Provide your reasoning, then clearly state your final prediction in this format:
Prediction: [your numeric prediction]%
"""
            prompt_path = save_dir / f"prompt_{save_str}_{i}_{self.model}.txt"
            with open(prompt_path, "w") as f:
                f.write(prompt)
            messages.append([
                {"role": "system", "content": "/nothink You are an expert programmer of the Gemmini deep learning accelerator."},
                {"role": "user", "content": prompt}
            ])

        # Read responses from save_dir if they exist
        responses = []
        responses_found = True
        for i in range(len(orig_codes)):
            response_path = save_dir / f"response_{save_str}_{i}_{self.model}.txt"
            if response_path.exists():
                with open(response_path, "r") as f:
                    responses.append(f.read())
            else:
                responses_found = False
            if responses_found:
                logger.info(f"Loaded evaluator responses from {save_dir}")
        
        scores = []
        if not responses_found:
            responses = []
            if "scratch" in self.model:
                temperature = 0.3
            else:
                temperature = 1
            chat_output = self.llm_client.chat_async(messages, num_candidates=1, temperature=temperature)
            for i, response_in_list in enumerate(chat_output):
                response = response_in_list[0]
                response_path = save_dir / f"response_{save_str}_{i}_{self.model}.txt"
                with open(response_path, "w") as f:
                    f.write(response)
                responses.append(response)

        for i, response in enumerate(responses):
            score = self._extract_score_robust(response)
            scores.append(score)
        
        return scores

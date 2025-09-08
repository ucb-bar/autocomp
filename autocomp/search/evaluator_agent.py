import re
import pathlib

from autocomp.common import LLMClient, logger
from autocomp.search.code_repo import CodeCandidate

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
        Robustly extract a score from LLM response using multiple fallback methods.
        Returns a float between 0 and 100, or 0.0 if parsing fails completely.
        """
        if not response or not isinstance(response, str):
            logger.warning("Empty or invalid response received")
            return 0.0
        
        response_clean = response.strip()
        
        # Method 1: Look for "Score:" followed by a number (most common format)
        score_patterns = [
            r"Score:\s*(\d+(?:\.\d+)?)",
            r"Score\s*:\s*(\d+(?:\.\d+)?)",
            r"score:\s*(\d+(?:\.\d+)?)",
            r"score\s*:\s*(\d+(?:\.\d+)?)",
            r"\*\*Score\*\*:\s*(\d+(?:\.\d+)?)",
        ]
        
        for pattern in score_patterns:
            matches = re.findall(pattern, response_clean, re.IGNORECASE)
            if matches:
                try:
                    score = float(matches[-1])  # Take the last match
                    if 0 <= score <= 100:
                        return score
                    # If score is out of range, clamp it
                    return max(0.0, min(100.0, score))
                except ValueError:
                    continue
        
        # Method 2: Look for standalone numbers that could be scores
        # Find all numbers in the response
        number_patterns = [
            r"\b(\d{1,3}(?:\.\d+)?)\b",  # 1-3 digit numbers (likely scores)
        ]
        
        for pattern in number_patterns:
            numbers = re.findall(pattern, response_clean)
            for num_str in reversed(numbers):  # Check from end backwards
                try:
                    num = float(num_str)
                    if 0 <= num <= 100:
                        logger.info(f"Extracted score {num} using fallback number pattern")
                        return num
                except ValueError:
                    continue
        
        # Method 3: Look for common score expressions
        expression_patterns = [
            r"(\d+(?:\.\d+)?)\s*(?:out of|/)\s*100",
            r"(\d+(?:\.\d+)?)\s*points?",
            r"(\d+(?:\.\d+)?)\s*%",
            r"rating.*?(\d+(?:\.\d+)?)",
            r"evaluate.*?(\d+(?:\.\d+)?)",
        ]
        
        for pattern in expression_patterns:
            matches = re.findall(pattern, response_clean, re.IGNORECASE)
            if matches:
                try:
                    score = float(matches[-1])
                    if 0 <= score <= 100:
                        logger.info(f"Extracted score {score} using expression pattern")
                        return score
                    # Convert percentage if needed
                    if score > 100:
                        score = score / 100.0 * 100.0
                        if 0 <= score <= 100:
                            return score
                except ValueError:
                    continue
        
        # Method 4: Try to find the last number in the response that could be a score
        all_numbers = re.findall(r"\d+(?:\.\d+)?", response_clean)
        if all_numbers:
            for num_str in reversed(all_numbers):
                try:
                    num = float(num_str)
                    if 0 <= num <= 100:
                        logger.warning(f"Using last valid number {num} as fallback score")
                        return num
                except ValueError:
                    continue
        
        # Method 5: Parse split method (original approach) as final fallback
        try:
            if "Score:" in response_clean:
                score_part = response_clean.split("Score:")[-1]
                # Try different ways to extract the number
                for line in score_part.split("\n"):
                    line = line.strip()
                    if line:
                        # Extract first number from the line
                        numbers = re.findall(r"\d+(?:\.\d+)?", line)
                        if numbers:
                            score = float(numbers[0])
                            if 0 <= score <= 100:
                                logger.warning(f"Using original split method, extracted {score}")
                                return score
        except (ValueError, IndexError):
            pass
        
        # If all methods fail, log the response and return 0
        logger.error(f"Failed to extract score from response: {response_clean[:200]}...")
        return 0.0

    def evaluate_plans(self, orig_codes: list[str], plans: list[str], save_dir: pathlib.Path) -> list[float]:
        messages = []
        for i, (orig_code, plan) in enumerate(zip(orig_codes, plans)):
            prompt = f"""You are an evaluator for a tensor processing optimization plan.
You will be given a code implementation of a tensor processing operation and a plan to optimize it.
Your task is to evaluate the plan and return a score between 0 (does not improve performance) and 100 (significant improvement).
The score should be based on the following criteria:
1. The plan is valid and can be applied to the code to improve performance.
2. The plan is efficient and reduces the execution time of the code.
3. The plan is optimal and there is no better plan that can be applied to the code to improve performance.

The code implementation is as follows:
{orig_code}

The plan is as follows:
{plan}

IMPORTANT: Please provide your reasoning, then clearly state your final score in this exact format:
Score: [your numeric score between 0 and 100]
"""
            prompt_path = save_dir / f"prompt_{i}_{self.model}.txt"
            with open(prompt_path, "w") as f:
                f.write(prompt)
            messages.append([{"role": "user", "content": prompt}])
        
        try:
            responses = self.llm_client.chat_async(messages, num_candidates=1, temperature=1)
            scores = []
            for i, response in enumerate(responses):
                response_path = save_dir / f"response_{i}_{self.model}.txt"
                with open(response_path, "w") as f:
                    f.write(response[0])
                score = self._extract_score_robust(response[0])
                scores.append(score)
                if score == 0.0:
                    logger.warning(f"Score extraction failed for plan {i}, using 0.0 as fallback")
            
            return scores
            
        except Exception as e:
            logger.error(f"Error in evaluate_plans: {e}")
            # Return zero scores as fallback
            return [0.0] * len(orig_codes)
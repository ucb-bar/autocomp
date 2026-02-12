from typing import List

from autocomp.search.prob import Prob

class EvalBackend:
    def __init__(self):
        pass

    def evaluate_code(self, prob: Prob, code_strs: list[str], simulator: str) -> List[dict]:
        pass

    def get_hw_feedback(self, prob: Prob, code_strs: list[str]) -> list[list[str]]:
        """Return per-candidate hardware feedback strings. Default: no feedback."""
        return [[] for _ in code_strs]

    def get_backend_specific_rules(self) -> list[str]:
        """Return backend-specific rule strings for LLM prompts. Default: no rules."""
        return []

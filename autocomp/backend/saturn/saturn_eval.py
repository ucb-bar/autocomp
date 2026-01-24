from typing import List
from autocomp.backend.hardware_backend import HardwareBackend
from autocomp.search.prob import Prob

class SaturnHardwareBackend(HardwareBackend):
    def evaluate_code(self, prob: Prob, code_strs: list[str], simulator: str) -> List[dict]:
        """
        Evaluate code candidates and return results.
        
        Args:
            prob: Problem instance containing metadata (prob_type, prob_id, etc.)
            code_strs: List of code strings to evaluate
            simulator: Simulator/evaluation method to use (e.g., "firesim", "spike")
        
        Returns:
            List of dictionaries, one per code_str, each containing:
            - "correct": bool - Whether the code passed correctness tests
            - Performance metrics (for all existing backends, this is "latency"), but new metrics can be added
        """
        results = []
        for code_str in code_strs:
            # 1. Prepare code for evaluation (may need to wrap in test harness)
            # 2. Run code through simulator/evaluator
            # 3. Parse results and extract metrics
            # 4. Run correctness tests
            result = {
                "correct": True,  # or False
                # Add your metrics here, e.g.:
                "latency": 123.45,
            }
            results.append(result)
        return results

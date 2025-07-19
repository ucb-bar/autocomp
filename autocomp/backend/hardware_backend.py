from typing import List

from autocomp.search.prob import Prob

class HardwareBackend:
    def __init__(self):
        pass

    def evaluate_code(self, prob: Prob, code_strs: list[str], simulator: str) -> List[dict]:
        pass

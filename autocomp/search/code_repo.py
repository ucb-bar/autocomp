import pathlib

from autocomp.common import logger

def copy_candidate(candidate: 'CodeCandidate') -> 'CodeCandidate':
    """
    Create a copy of a CodeCandidate object.
    """
    new_candidate = CodeCandidate(
        parent=candidate.parent,
        plan=candidate.plan,
        code=candidate.code,
        score=candidate.score,
        spad_acc_stats=candidate.spad_acc_stats[:],  # Copy the spad_acc_stats list
        plan_gen_model=candidate.plan_gen_model,
        code_gen_model=candidate.code_gen_model
    )
    return new_candidate

class CodeCandidate:
    parent: 'CodeCandidate'
    plan: str | None
    score: float | None
    implemented: bool
    code: str | None
    """
    Represents a single version of the code with an associated optimization plan.
    """
    def __init__(self, parent: 'CodeCandidate', plan: str, code: str, score: float=None, spad_acc_stats: list[str]=None,
                 plan_gen_model=None, code_gen_model=None):
        self.parent = parent # Pointer to parent CodeCandidate
        self.plan = plan
        self.score = score  # Score based on the evaluation function
        if not code:
            self.implemented = False  # Whether the code has been implemented
            self.code = None
        else:
            self.implemented = True
            self.code = code

        if spad_acc_stats is None:
            self.spad_acc_stats = list()
        else:
            self.spad_acc_stats = spad_acc_stats # spad_acc_stats to pass to the next iteration

        self.plan_gen_model = plan_gen_model
        self.code_gen_model = code_gen_model

    def __repr__(self):
        repr_str = f"CodeCandidate(parent={repr(self.parent)},\nplan="
        if self.plan is None:
            repr_str += "None"
        else:
            repr_str += f"'''{self.plan}'''"
        repr_str += f",\ncode='''{self.code}''',\nscore={self.score},\nspad_acc_stats={repr(self.spad_acc_stats)},\nplan_gen_model='{self.plan_gen_model}',\ncode_gen_model='{self.code_gen_model}')"
        return repr_str

    def update_spad_acc_stats(self, spad_acc_stats: list[str]) -> None:
        self.spad_acc_stats.extend(spad_acc_stats)

class CodeRepository:
    """
    Stores multiple code candidates per iteration.
    """
    def __init__(self):
        self.candidates_per_iteration: list[list[CodeCandidate]] = []  # List of lists of CodeCandidates per iteration
        self.other_candidates: dict[list[CodeCandidate]] = {} # Repository for holding other candidates of interest

    def add_candidates(self, candidates: list, iteration: int):
        """Add a set of candidates for a given iteration."""
        if not isinstance(iteration, int):
            if iteration not in self.other_candidates:
                self.other_candidates[iteration] = []
            self.other_candidates[iteration].extend(candidates)
            return
        
        if iteration >= len(self.candidates_per_iteration):
            self.candidates_per_iteration.append(candidates[:])
        else:
            self.candidates_per_iteration[iteration].extend(candidates)

    def get_candidates(self, iteration: int) -> list:
        """Retrieve all candidates for a given iteration."""
        if isinstance(iteration, int):
            return self.candidates_per_iteration[iteration]
        else:
            return self.other_candidates[iteration]

    def display_latest_candidates(self):
        """Display the code of the latest candidates."""
        if self.candidates_per_iteration:
            for candidate in self.candidates_per_iteration[-1]:
                logger.info(repr(candidate))

    def save_candidates(self, iteration: int, save_dir: pathlib.Path):
        """Save the code of the candidates for a given iteration."""
        for c_i, candidate in enumerate(self.get_candidates(iteration)):
            path = save_dir / f"candidate_{c_i}.txt"
            with open(path, "w") as f:
                f.write(repr(candidate))
            logger.debug("Saved candidate code to %s", path)

    def load_candidates(self, iteration: int, save_dir: pathlib.Path):
        """Save the code of the candidates for a given iteration."""
        candidate_paths = save_dir.glob("candidate_*.txt")
        candidates = []
        for path in candidate_paths:
            cand = eval(path.read_text())
            logger.debug("Loaded candidate from %s", path)
            candidates.append(cand)
        self.add_candidates(candidates, iteration)
        return len(candidates)
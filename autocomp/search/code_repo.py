import pathlib
import os

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
        plan_gen_model=candidate.plan_gen_model,
        code_gen_model=candidate.code_gen_model,
        stdout=candidate.stdout,
        stderr=candidate.stderr
    )
    new_candidate.translation_score = candidate.translation_score
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
    def __init__(self, parent: 'CodeCandidate', plan: str, code: str, score: float=None, translation_score: float=None, back: list[str]=None,
                 plan_gen_model=None, code_gen_model=None, stdout: str=None, stderr: str=None):
        self.parent = parent # Poisnter to parent CodeCandidate
        self.plan = plan
        self.score = score  # Score based on the evaluation function
        self.translation_score = translation_score
        self.back = back or []  # Used by __repr__; keep always defined for cached IO
        if not code:
            self.implemented = False  # Whether the code has been implemented
            self.code = None
        else:
            self.implemented = True
            self.code = code

        self.plan_gen_model = plan_gen_model
        self.code_gen_model = code_gen_model
        self.stdout = stdout
        self.stderr = stderr

    def __repr__(self):
        repr_str = f"CodeCandidate(parent={repr(self.parent)},\nplan="
        if self.plan is None:
            repr_str += "None"
        else:
            escaped_plan = self.plan.replace('\'', '\\\'')
            repr_str += f"'''{escaped_plan}'''"
        code_str = self.code if self.code is not None else ""
        escaped_code = code_str.replace('\'', '\\\'')
        repr_str += f",\ncode='''{escaped_code}''',\nscore={self.score},\ntranslation_score={self.translation_score},\nback={repr(self.back)},\nplan_gen_model='{self.plan_gen_model}',\ncode_gen_model='{self.code_gen_model}',\nstdout={repr(self.stdout)},\nstderr={repr(self.stderr)})"
        return repr_str 

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

    def get_candidates(self, iteration: int) -> list[CodeCandidate]:
        """Retrieve all candidates for a given iteration."""
        if isinstance(iteration, int):
            return self.candidates_per_iteration[iteration]
        else:
            return self.other_candidates[iteration]

    def display_latest_candidates(self) -> None:
        """Display the code of the latest candidates."""
        if self.candidates_per_iteration:
            for candidate in self.candidates_per_iteration[-1]:
                logger.info(repr(candidate))

    def save_candidates(self, iteration: int, save_dir: pathlib.Path) -> None:
        """Save the code of the candidates for a given iteration."""
        for c_i, candidate in enumerate(self.get_candidates(iteration)):
            path = save_dir / f"candidate_{c_i}.txt"
            # Use UTF-8 explicitly to avoid Windows cp1252 encoding errors.
            with open(path, "w", encoding="utf-8", errors="replace") as f:
                f.write(repr(candidate))
            logger.debug("Saved candidate code to %s", path)

    def load_candidates(self, iteration: int, save_dir: pathlib.Path) -> int:
        """Load the code of the candidates to the repository for a given iteration."""
        candidate_paths = save_dir.glob("candidate_*.txt")
        candidates = []
        for path in candidate_paths:
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
                if not raw.strip():
                    logger.warning("Skipping empty candidate cache file: %s", path)
                    continue
                cand = eval(raw)
                if cand is None:
                    logger.warning("Skipping candidate cache file loaded as None: %s", path)
                    continue
                logger.debug("Loaded candidate from %s", path)
                candidates.append(cand)
            except SyntaxError as e:
                # Likely a partially written/corrupted file.
                logger.error("Skipping corrupted candidate cache file %s: %s", path, e)
                continue
            except Exception as e:
                logger.error("Error loading candidate from %s: %s", path, e)
                raise e
        self.add_candidates(candidates, iteration)
        return len(candidates)
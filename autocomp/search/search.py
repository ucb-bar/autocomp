import pathlib
import random
import json
import time
from pathlib import Path

import wandb

from autocomp.common import logger, SOLS_DIR, REPO_ROOT
from autocomp.common.llm_utils import aggregate_usage
from autocomp.search.code_repo import CodeCandidate, CodeRepository
from autocomp.search.prob import Prob
from autocomp.agents.llm_ensemble import LLMEnsemble
from autocomp.backend.eval_backend import EvalBackend

# Register LLM agents
from autocomp.agents.gemmini.gemmini_agent import GemminiLLMAgent
from autocomp.agents.cuda.cuda_agent import CudaLLMAgent
from autocomp.agents.trn_nki1.trn_nki1_agent import TrnNki1LLMAgent
from autocomp.agents.trn_nki2.trn_nki2_agent import TrnNki2LLMAgent
from autocomp.agent_builder.built_agent import BuiltLLMAgent
from autocomp.agents.saturn.saturn_agent import SaturnLLMAgent

# ... register more LLM agents here ...
# Register eval backends
from autocomp.backend.gemmini.gemmini_eval import GemminiEvalBackend
from autocomp.backend.kernelbench.kb_eval import KBEvalBackend, KERNELBENCH_DIR
from autocomp.backend.gpumode.gpumode_eval import GpuModeEvalBackend
from autocomp.backend.trn.trn_eval import TrnEvalBackend
from autocomp.backend.tpu.tpu_eval import TpuEvalBackend
from autocomp.backend.jaxbench.jaxbench_eval import JaxBenchEvalBackend
from autocomp.backend.saturn.saturn_eval import SaturnEvalBackend
from autocomp.backend.xnnpack.xnnpack_eval import XnnpackEvalBackend
# ... register more eval backends here ...
# Hardware configs
from autocomp.hw_config import (
    CudaHardwareConfig,
    GemminiHardwareConfig,
    TrnHardwareConfig,
    TpuHardwareConfig,
)  # noqa: F401 — re-exported for backwards compat


def create_backend_and_agents(
    backend_name: str,
    agent_name: str,
    hw_config,
    prob: Prob,
    models: list,
    code_models: list = None,
    menu_strategy: str = None,
    fine_grained_isa: bool = False,
    example_rate: float = 0.0,
    cache_dir=None,
):
    """Create eval backend and agent ensembles.

    Args:
        backend_name: Which eval backend to use.
        agent_name: Which agent type. Defaults based on backend_name (kernelbench/gpumode -> cuda).
        hw_config: A HardwareConfig instance describing the target hardware.
        prob: The problem to optimize.
        models: List of model strings for planning.
        code_models: Optional list of model strings for code implementation.
        cache_dir: Optional directory for agent caches (e.g. ISA/example selection).
    """
    # Create eval backend
    if backend_name == "kernelbench":
        eval_backend = KBEvalBackend()
    elif backend_name == "gpumode":
        eval_backend = GpuModeEvalBackend()
    elif backend_name == "gemmini":
        eval_backend = GemminiEvalBackend(hw_config)
    elif backend_name == "trn":
        eval_backend = TrnEvalBackend()
    elif backend_name == "tpu":
        eval_backend = TpuEvalBackend()
    elif backend_name == "jaxbench":
        eval_backend = JaxBenchEvalBackend()
    elif backend_name == "saturn":
        eval_backend = SaturnEvalBackend()
    elif backend_name == "xnnpack":
        eval_backend = XnnpackEvalBackend()
    else:
        raise ValueError(f"Unknown backend: {backend_name}")

    # Create agents
    if agent_name == "cuda":
        agent = LLMEnsemble([CudaLLMAgent(m, hw_config, eval_backend) for m in models])
        code_agent = (
            LLMEnsemble([CudaLLMAgent(m, hw_config, eval_backend) for m in code_models])
            if code_models
            else None
        )
    elif agent_name == "gemmini":
        agent = LLMEnsemble(
            [GemminiLLMAgent(m, hw_config, eval_backend) for m in models]
        )
        code_agent = (
            LLMEnsemble(
                [GemminiLLMAgent(m, hw_config, eval_backend) for m in code_models]
            )
            if code_models
            else None
        )
    elif agent_name == "trn" or agent_name == "trn-nki1":
        agent = LLMEnsemble(
            [TrnNki1LLMAgent(m, hw_config, eval_backend) for m in models]
        )
        code_agent = (
            LLMEnsemble(
                [TrnNki1LLMAgent(m, hw_config, eval_backend) for m in code_models]
            )
            if code_models
            else None
        )
    elif agent_name == "trn-nki2":
        agent = LLMEnsemble(
            [TrnNki2LLMAgent(m, hw_config, eval_backend) for m in models]
        )
        code_agent = (
            LLMEnsemble(
                [TrnNki2LLMAgent(m, hw_config, eval_backend) for m in code_models]
            )
            if code_models
            else None
        )
    elif agent_name.startswith("built:") or Path(agent_name).is_dir():
        # "built:<name>" resolves to .built/<name>/; direct paths also accepted
        _BUILT_DIR = REPO_ROOT / "autocomp" / "agent_builder" / ".built"
        if agent_name.startswith("built:"):
            built_name = agent_name[len("built:") :]
            config_dir = _BUILT_DIR / built_name
        else:
            config_dir = Path(agent_name)
        if not config_dir.is_dir():
            available = (
                [p.parent.name for p in _BUILT_DIR.glob("*/agent_config.yaml")]
                if _BUILT_DIR.is_dir()
                else []
            )
            raise ValueError(
                f"Built agent config not found at '{config_dir}'. "
                f"Available: {available}"
            )
        logger.info("Using built agent from %s", config_dir)
        agent = LLMEnsemble(
            [
                BuiltLLMAgent(
                    m,
                    config_dir,
                    hw_config,
                    eval_backend,
                    menu_strategy,
                    fine_grained_isa=fine_grained_isa,
                    example_rate=example_rate,
                    cache_dir=cache_dir,
                )
                for m in models
            ]
        )
        code_agent = (
            LLMEnsemble(
                [
                    BuiltLLMAgent(
                        m,
                        config_dir,
                        hw_config,
                        eval_backend,
                        menu_strategy,
                        fine_grained_isa=fine_grained_isa,
                        example_rate=example_rate,
                        cache_dir=cache_dir,
                    )
                    for m in code_models
                ]
            )
            if code_models
            else None
        )
    else:
        raise ValueError(
            f"Unknown agent name: '{agent_name}'. Use 'cuda', 'gemmini', 'trn', 'built:<name>', or a path to a built agent directory."
        )

    return eval_backend, agent, code_agent


def load_initial_code(backend_name: str, prob: "Prob") -> str:
    """Load initial code for the given backend and problem.

    Also sets ``prob.sol_file`` to the resolved source-file path.
    """
    if prob.sol_file:
        return prob.sol_file.read_text()

    prob_type, prob_id = prob.prob_type, prob.prob_id

    if backend_name == "kernelbench":
        if "kb-" in prob_type:
            level_str = prob_type.split("-")[1]
            kb_level_dir = pathlib.Path(KERNELBENCH_DIR) / "KernelBench" / level_str
            matches = list(kb_level_dir.glob(f"{prob_id}_*.py"))
            if not matches:
                raise FileNotFoundError(
                    f"No file matching {prob_id}_*.py in {kb_level_dir}"
                )
            prob.sol_file = matches[0]
            return matches[0].read_text().replace("Model", "ModelNew")
    elif backend_name == "gpumode":
        sol_dir = SOLS_DIR / prob_type
        matches = list(sol_dir.glob(f"{prob_id}_*.py"))
        if not matches:
            raise FileNotFoundError(f"No file matching {prob_id}_*.py in {sol_dir}")
        prob.sol_file = matches[0]
        return matches[0].read_text()
    elif backend_name == "gemmini":
        if "admm" in prob_type:
            sol_path = SOLS_DIR / "admm-multifunction" / f"sol{prob_id}_unopt_sw.c"
        else:
            sol_path = SOLS_DIR / prob_type / f"sol{prob_id}_exo_baseline.c"
        prob.sol_file = sol_path
        return sol_path.read_text()
    elif backend_name == "trn":
        sol_dir = SOLS_DIR / prob_type
        matches = list(sol_dir.glob(f"{prob_id}_*.py"))
        if not matches:
            raise FileNotFoundError(f"No file matching {prob_id}_*.py in {sol_dir}")
        prob.sol_file = matches[0]
        return matches[0].read_text()
    elif backend_name == "tpu":
        sol_dir = SOLS_DIR / prob_type
        matches = list(sol_dir.glob(f"{prob_id}_*.py"))
        if not matches:
            raise FileNotFoundError(f"No file matching {prob_id}_*.py in {sol_dir}")
        prob.sol_file = matches[0]
        return matches[0].read_text()
    elif backend_name == "jaxbench":
        sol_dir = SOLS_DIR / prob_type
        matches = list(sol_dir.glob(f"{prob_id}_*.py"))
        if matches:
            prob.sol_file = matches[0]
            return matches[0].read_text()
        from autocomp.backend.jaxbench.jaxbench_eval import extract_workload_code

        return extract_workload_code(prob)
    elif backend_name == "saturn":
        sol_dir = SOLS_DIR / prob_type
        matches = list(sol_dir.glob(f"{prob_id}_*.c"))
        if not matches:
            raise FileNotFoundError(f"No file matching {prob_id}_*.c in {sol_dir}")
        with open(matches[0]) as f:
            return f.read()
    elif backend_name == "xnnpack":
        sol_dir = SOLS_DIR / prob_type
        matches = list(sol_dir.glob(f"{prob_id}_*.c"))
        if not matches:
            raise FileNotFoundError(f"No file matching {prob_id}_*.c in {sol_dir}")
        with open(matches[0]) as f:
            return f.read()
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


def _find_latest_candidates_dir(output_dir: pathlib.Path) -> pathlib.Path | None:
    """Find the highest-numbered candidates-iter-N directory in an output directory."""
    best_iter = -1
    best_dir = None
    for d in output_dir.glob("candidates-iter-*"):
        if d.is_dir():
            try:
                n = int(d.name.split("-")[-1])
                if n > best_iter:
                    best_iter = n
                    best_dir = d
            except ValueError:
                continue
    return best_dir


class SearchStrategy:
    """
    Base class for different search strategies.
    """

    def __init__(
        self,
        output_dir: pathlib.Path,
        eval_backend: EvalBackend,
        agent: LLMEnsemble,
        orig_code: str,
        prob: Prob,
        metric: str,
        simulator: str,
        give_score_feedback: float,
        give_util_feedback: float,
        give_hw_feedback: float,
        include_ancestors: bool,
        plan_icl_examples: bool,
        code_icl_examples: bool,
        dropout_menu_options: float,
        prevent_duplicate_level: int,
        translate_iters: int,
        translate_perf_threshold: float,
        translate_drop_original: bool,
        translate_score: bool,
        code_agent: LLMEnsemble = None,
        early_stop_iters: int = 0,
        early_stop_threshold: float = 1.0,
        continue_from: str | pathlib.Path | None = None,
        use_edits: bool = False,
    ):
        self.repository = CodeRepository()  # Stores the code candidates
        self.agent = agent  # The agent used to propose optimizations (planning)
        self.code_agent = (
            code_agent if code_agent is not None else agent
        )  # The agent used for code implementation
        self.use_edits = use_edits
        self.prob = prob
        self.problem = prob.name if hasattr(prob, "name") else str(prob)
        self.plan_models = sorted(
            {a.llm_client.provider + "::" + a.llm_client.model for a in self.agent.llms}
        )
        self.code_models = sorted(
            {
                a.llm_client.provider + "::" + a.llm_client.model
                for a in self.code_agent.llms
            }
        )
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.eval_backend = eval_backend
        self.metric = metric
        self.simulator = simulator
        self.give_score_feedback = give_score_feedback
        self.give_util_feedback = give_util_feedback
        self.give_hw_feedback = give_hw_feedback
        self.include_ancestors = include_ancestors
        self.plan_icl_examples = plan_icl_examples
        self.code_icl_examples = code_icl_examples
        self.dropout_menu_options = dropout_menu_options
        self.prevent_duplicate_level = prevent_duplicate_level
        self.translate_iters = translate_iters
        self.translate_perf_threshold = translate_perf_threshold
        self.translate_drop_original = translate_drop_original
        self.translate_score = translate_score
        self.early_stop_iters = early_stop_iters
        self.early_stop_threshold = early_stop_threshold

        save_dir = self.output_dir / f"candidates-iter-0"
        save_dir.mkdir(parents=True, exist_ok=True)
        num_cands_loaded = self.repository.load_candidates(0, save_dir)
        if num_cands_loaded > 0:
            logger.info("Loaded initial code from %s", save_dir)
        elif continue_from:
            continue_dir = pathlib.Path(continue_from)
            src_dir = _find_latest_candidates_dir(continue_dir)
            if src_dir is None:
                raise ValueError(
                    f"No candidates-iter-* directories found in {continue_dir}"
                )
            logger.info("Continuing from %s", src_dir)
            import shutil

            for f in src_dir.glob("candidate_*.txt"):
                shutil.copy2(f, save_dir / f.name)
            num_cands_loaded = self.repository.load_candidates(0, save_dir)
            if num_cands_loaded == 0:
                raise ValueError(f"No candidates loaded from {src_dir}")
            logger.info(
                "Loaded %d candidates from previous run %s", num_cands_loaded, src_dir
            )
        else:
            orig_code_candidate = CodeCandidate(None, None, orig_code)
            eval_save_dir = self.output_dir / "eval-results-iter-0"
            eval_save_dir.mkdir(parents=True, exist_ok=True)
            self.evaluate_candidates(
                [orig_code_candidate], self.metric, save_dir=eval_save_dir
            )  # Evaluate the initial code
            if orig_code_candidate.score == float("inf"):
                if orig_code_candidate.stderr:
                    logger.error("Initial code failed with error: %s", orig_code_candidate.stderr)
                logger.error("Initial code failure details saved to %s", eval_save_dir)
                raise ValueError("Initial code is incorrect.")
            self.add_feedback([orig_code_candidate])
            self.repository.add_candidates(
                [orig_code_candidate], 0
            )  # Add the initial code as the first candidate
            self.repository.save_candidates(0, save_dir)
        initial_code_candidates: list[CodeCandidate] = self.repository.get_candidates(0)
        logger.info("Initial code scores:")
        for candidate in initial_code_candidates:
            logger.info(candidate.score)

        self._save_run_metadata()

    def _save_run_metadata(self):
        """Save run configuration metadata to a JSON file in the output directory."""
        serializable = (str, int, float, bool, list, tuple, type(None))
        metadata = {
            k: v
            for k, v in vars(self).items()
            if not k.startswith("_") and isinstance(v, serializable)
        }
        try:
            with open(self.output_dir / "run_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2, default=str)
        except Exception as e:
            logger.warning("Failed to save run metadata: %s", e)

    def _get_best_candidate(self):
        """Find the global best candidate across all iterations."""
        best = None
        for candidates in self.repository.candidates_per_iteration:
            for c in candidates:
                if c.score is not None and c.score != float("inf"):
                    if best is None or c.score < best.score:
                        best = c
        return best

    def _save_best_candidate(self):
        """Find the global best candidate and write its source code to disk."""
        best = self._get_best_candidate()
        if best is None:
            return None
        ext = self.prob.sol_file.suffix if self.prob.sol_file else ".txt"
        path = self.output_dir / f"best_candidate_so_far{ext}"
        try:
            path.write_text(best.code)
            logger.info("Best candidate score: %s (saved to %s)", best.score, path.name)
        except Exception as e:
            logger.warning("Failed to save best candidate: %s", e)
        return best

    def propose_optimizations_iter(
        self, candidates: list[CodeCandidate], num_plans: int
    ) -> list[CodeCandidate]:
        """
        Use the LLM to propose new optimization plans for the given code.
        """
        raise NotImplementedError

    def evaluate_candidates(
        self,
        candidates: list[CodeCandidate],
        metric: str,
        cur_iter: int = None,
        save_dir: pathlib.Path = None,
    ) -> list[CodeCandidate]:
        """
        Evaluate the candidates based on the provided optimization metric
        and update their scores.
        """
        # Load stats if they already exist in the save_dir
        if (
            save_dir is not None
            and save_dir.exists()
            and all(
                (save_dir / f"code_{i}_result.txt").exists()
                for i in range(len(candidates))
            )
        ):
            logger.info(
                f"Loading cached evaluation results for all {len(candidates)} candidates from {save_dir}"
            )
            per_cand_stats = []
            for i in range(len(candidates)):
                with open(save_dir / f"code_{i}_result.txt", "r") as f:
                    per_cand_stats.append(json.load(f))
        else:
            per_cand_stats = self.eval_backend.evaluate_code(
                self.prob, [candidate.code for candidate in candidates], self.simulator
            )

            # Save stats
            if save_dir is not None:
                for cand_i, stats in enumerate(per_cand_stats):
                    with open(save_dir / f"code_{cand_i}_result.txt", "w") as f:
                        json.dump(stats, f, indent=4)
                    with open(save_dir / f"code_{cand_i}_result_full.txt", "w") as f:
                        f.write(str(stats).replace("\\n", "\n"))
                        if candidates[cand_i].parent is not None:
                            f.write(
                                "\nPrev latency: "
                                + str(candidates[cand_i].parent.score)
                                + "\n"
                            )
                        if stats["correct"]:
                            f.write("New latency: " + str(stats[metric]) + "\n")
                        else:
                            f.write("New latency: N/A\n")
                        plan_text = candidates[cand_i].plan
                        f.write(
                            "Plan: "
                            + (plan_text.replace("\\n", "\n") if plan_text is not None else "N/A")
                            + "\n"
                        )
                        f.write("\n" + repr(candidates[cand_i]))

        for cand_i, stats in enumerate(per_cand_stats):
            if stats["correct"]:
                # Assume the metric exists if the code passed tests
                candidates[cand_i].score = stats[metric]
            else:
                candidates[cand_i].score = float("inf")
            # Store stdout and stderr for failed candidates
            if "stdout" in stats:
                candidates[cand_i].stdout = stats["stdout"]
            if "stderr" in stats:
                candidates[cand_i].stderr = stats["stderr"]
        return candidates

    def _score_translation(
        self,
        candidates: list[CodeCandidate],
        save_dir: pathlib.Path,
    ) -> None:
        """Score translation completeness, with per-directory caching like evaluate_candidates."""
        cache_path = save_dir / "translation_scores.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            scores = [float(s) for s in cached]
            if len(scores) == len(candidates):
                logger.info("Loading cached translation scores from %s", cache_path)
                for cand, ts in zip(candidates, scores):
                    cand.translation_score = ts
                return
            logger.warning(
                "Cached translation scores length mismatch (%d vs %d), recomputing",
                len(scores), len(candidates),
            )

        original_code = self.repository.get_candidates(0)[0].code
        scores = self.agent.score_translation_completeness(
            original_code, candidates, prob=self.prob
        )
        for cand, ts in zip(candidates, scores):
            cand.translation_score = ts

        save_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(scores))

    def _check_translation_complete(self, beam_candidates: list[CodeCandidate], cur_iter: int) -> None:
        """End translation early if all beam candidates are fully translated."""
        if not self.translate_score or cur_iter >= self.translate_iters:
            return
        beam_scores = [
            c.translation_score for c in beam_candidates
            if c.translation_score is not None
        ]
        if beam_scores and min(beam_scores) >= 9.0:
            logger.info(
                "All %d beam candidates have translation score >= 9.0 (min=%.1f) — "
                "ending translation early (iter %d of %d).",
                len(beam_scores), min(beam_scores), cur_iter, self.translate_iters,
            )
            self.translate_iters = cur_iter

    def add_feedback(self, candidates: list[CodeCandidate]) -> list[CodeCandidate]:
        # NOTE: Assumes that feedback for a particular candidate only gets added once and does not get updated
        if self.give_hw_feedback <= 0:
            return candidates
        for cand_i in range(len(candidates)):
            if candidates[cand_i].hw_feedback:  # If already has feedback, skip
                continue
            feedback_per_cand = self.eval_backend.get_hw_feedback(
                self.prob, [candidates[cand_i].code]
            )
            feedback = feedback_per_cand[0]
            if feedback:
                logger.debug("Adding feedback to candidate %d: %s", cand_i, feedback)
                candidates[cand_i].update_hw_feedback(feedback)
        return candidates

    def filter_code_candidates(
        self,
        code_candidates: list[CodeCandidate],
        num_to_keep: int | None = None,
        cur_iter: int = 0,
        num_iters: int = 0,
    ) -> list:
        """
        Filter and return the top N code candidates based on their score.
        If N (num_to_keep) is not provided, will return all candidates with a higher score than their parent.
        """
        # Filter out incorrect candidates
        code_candidates = [c for c in code_candidates if c.score != float("inf")]

        keep_factor = 1
        if cur_iter <= self.translate_iters:
            keep_factor = self.translate_perf_threshold
        elif cur_iter <= 2:
            keep_factor = 1.1
        if num_to_keep is None:
            cur_candidates = []
            for c in code_candidates:
                if isinstance(c.parent, list):
                    if all([c.score < p.score for p in c.parent]):
                        cur_candidates.append(c)
                else:
                    if c.score < c.parent.score * keep_factor:
                        if c.score >= c.parent.score:
                            logger.debug(
                                f"Keep factor used at iter {cur_iter}, candidate\n{c}"
                            )
                        cur_candidates.append(c)
        else:
            cur_candidates = []
            use_translation_sort = (
                self.translate_score and cur_iter <= self.translate_iters
            )
            if use_translation_sort:
                code_candidates.sort(
                    key=lambda c: (-(c.translation_score or 0), c.score),
                )
            else:
                code_candidates.sort(key=lambda c: c.score, reverse=False)
            for cand in code_candidates:
                if len(cur_candidates) >= num_to_keep:
                    break
                if cand.parent is None:
                    cur_candidates.append(cand)
                    continue
                if not use_translation_sort:
                    # Don't keep any candidates with a score same or higher than their parent
                    if cand.score >= cand.parent.score * keep_factor:
                        logger.debug(
                            f"Working candidate has score {cand.score} >= parent score {cand.parent.score}."
                        )
                        logger.debug(f"Candidate plan:\n{cand.plan}")
                        logger.debug(f"Candidate code:\n{cand.code}")
                        continue
                dont_add = False
                if self.prevent_duplicate_level < 0:
                    pass
                elif self.prevent_duplicate_level == 2:
                    # Don't keep any candidates with parents in common other than the root candidate
                    # And also don't keep the parents of candidates already in the list
                    for already_in_cand in cur_candidates:
                        if dont_add == True:
                            break
                        if (cand.parent == already_in_cand.parent) and (
                            cand.plan == already_in_cand.plan
                        ):
                            dont_add = True
                            break
                        already_in_parent_cand = already_in_cand.parent
                        while already_in_parent_cand is not None:
                            if (
                                cand == already_in_parent_cand
                                or cand.parent == already_in_parent_cand
                            ) and (already_in_parent_cand.parent is not None):
                                dont_add = True
                                break
                            already_in_parent_cand = already_in_parent_cand.parent
                elif self.prevent_duplicate_level == 1:
                    # Don't keep any candidates with same parent as one already in the list, unless that parent is the root candidate
                    for already_in_cand in cur_candidates:
                        if (
                            cand.parent == already_in_cand.parent
                            and cand.parent.parent is not None
                        ):
                            dont_add = True
                else:
                    # Don't keep any candidates with same parent and plan as one already in the list (the one already in the list is equal or better)
                    for already_in_cand in cur_candidates:
                        if (cand.parent == already_in_cand.parent) and (
                            cand.plan == already_in_cand.plan
                        ):
                            dont_add = True

                if not dont_add:
                    cur_candidates.append(cand)

        return cur_candidates

    def init_wandb(self):
        # start a new wandb run to track this script
        wandb.init(
            entity=None,
            # set the wandb project where this run will be logged
            project=None,
            # track hyperparameters and run metadata
            config=vars(self),
        )
        logger.info("Initialized wandb run, id: %s", wandb.run.name)

    def should_early_stop(self, losses: list[float], cur_iter: int) -> bool:
        if self.early_stop_iters <= 0 or cur_iter < self.early_stop_iters + 1:
            return False
        if cur_iter <= self.translate_iters + self.early_stop_iters:
            return False
        old_loss = losses[cur_iter - self.early_stop_iters - 1]
        new_loss = losses[cur_iter - 1]
        if old_loss == 0:
            return False
        ratio = new_loss / old_loss
        if ratio >= self.early_stop_threshold:
            logger.info(
                "Early stopping: no improvement over %d iters (ratio %.4f >= %.4f)",
                self.early_stop_iters,
                ratio,
                self.early_stop_threshold,
            )
            return True
        return False


class ExhaustiveSearchStrategy(SearchStrategy):
    """
    Tries every optimization menu option exhaustively, generating plans and
    implementations in parallel batches.
    """

    def __init__(
        self, *args, plans_per_option: int = 1, num_code_candidates: int = 1, **kwargs
    ):
        self.plans_per_option = plans_per_option
        self.num_code_candidates = num_code_candidates
        super().__init__(*args, **kwargs)

    def propose_optimizations_iter(
        self,
        parent_candidates: list[CodeCandidate],
        save_dir: pathlib.Path,
        cur_iter: int = None,
        num_iters: int = None,
    ) -> list[CodeCandidate]:
        menu_options = list(range(1, len(self.agent.get_opt_menu_options()) + 1))

        duplicated_candidates: list[CodeCandidate] = []
        duplicated_save_strs: list[str] = []
        duplicated_force_opt_menu: list[int] = []
        for p_i, parent_cand in enumerate(parent_candidates):
            for menu_opt in menu_options:
                duplicated_candidates.append(parent_cand)
                duplicated_save_strs.append(f"parent{p_i}_opt{menu_opt}")
                duplicated_force_opt_menu.append(menu_opt)

        return self.agent.propose_optimizations_parallel(
            candidate_lst=duplicated_candidates,
            num_plans=self.plans_per_option,
            save_dir=save_dir,
            save_strs=duplicated_save_strs,
            prob=self.prob,
            force_opt_menu_lst=duplicated_force_opt_menu,
            give_score_feedback=self.give_score_feedback,
            give_util_feedback=self.give_util_feedback,
            give_hw_feedback=self.give_hw_feedback,
            include_ancestors=self.include_ancestors,
            plan_icl_examples=self.plan_icl_examples,
            dropout_menu_options=self.dropout_menu_options,
            cur_iter=cur_iter,
            num_iters=num_iters,
        )

    def optimize(self, iterations: int):
        """Run the optimization process with the selected search strategy for multiple iterations."""
        losses = []
        for i in range(1, iterations + 1):
            logger.info(f"Iteration {i} of optimization:")

            # Get current candidates (use initial code if it's the first iteration)
            cur_cand_idx = i - 1
            current_candidates = self.repository.get_candidates(cur_cand_idx)
            while len(current_candidates) == 0:
                logger.warning(
                    "No candidates found for iteration %d. Trying candidates from iteration %d.",
                    cur_cand_idx,
                    cur_cand_idx - 1,
                )
                cur_cand_idx -= 1
                current_candidates = self.repository.get_candidates(cur_cand_idx)

            cur_cand_scores = [cand.score for cand in current_candidates]
            best_loss = min(cur_cand_scores)
            losses.append(best_loss)

            if self.should_early_stop(losses, i):
                break

            # Step 1: Propose optimizations for each candidate x each menu option
            save_dir = self.output_dir / f"generated-plans-iter-{i}"
            save_dir.mkdir(parents=True, exist_ok=True)
            plan_only_candidates = self.propose_optimizations_iter(
                current_candidates, save_dir, cur_iter=i, num_iters=iterations
            )
            logger.info(f"Proposed {len(plan_only_candidates)} new optimizations.")

            # Step 2: Generate code implementations in parallel
            save_dir = self.output_dir / f"generated-code-iter-{i}"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_strs = [f"plan{p_i}" for p_i in range(len(plan_only_candidates))]
            if self.use_edits:
                impl_candidates = self.code_agent.implement_code_edits_parallel(
                    plan_only_candidates,
                    self.num_code_candidates,
                    save_dir,
                    save_strs=save_strs,
                    prob=self.prob,
                )
            else:
                impl_candidates = self.code_agent.implement_code_parallel(
                    plan_only_candidates,
                    self.num_code_candidates,
                    save_dir,
                    save_strs=save_strs,
                    prob=self.prob,
                )
            logger.info(f"Generated {len(impl_candidates)} implementations from {len(plan_only_candidates)} plans.")

            # Step 3: Evaluate the generated implementations
            save_dir = self.output_dir / f"eval-results-iter-{i}"
            save_dir.mkdir(parents=True, exist_ok=True)
            evaluated_code_candidates = self.evaluate_candidates(
                impl_candidates, metric=self.metric, cur_iter=i, save_dir=save_dir
            )
            logger.info(f"Evaluated {len(evaluated_code_candidates)} implementations.")

            # Step 4: Filter and rank the implementations
            improving_candidates = self.filter_code_candidates(
                evaluated_code_candidates, cur_iter=i, num_iters=iterations
            )
            candidates_for_next_iter = self.filter_code_candidates(
                improving_candidates, num_to_keep=1, cur_iter=i, num_iters=iterations
            )

            # Step 5: Save the improving candidates and update the repository
            self.repository.add_candidates(improving_candidates, "improving")
            self.repository.add_candidates(candidates_for_next_iter, i)
            logger.info(
                f"Filtered down to {len(candidates_for_next_iter)} code candidates."
            )
            logger.info(
                f"Saved {len(improving_candidates)} improving code candidates to repository."
            )

            # Step 6: Save the latest candidates to disk
            save_dir = self.output_dir / f"candidates-iter-{i}"
            save_dir.mkdir(parents=True, exist_ok=True)
            self.repository.save_candidates(i, save_dir)
            logger.info("New candidate scores:")
            for candidate in candidates_for_next_iter:
                logger.info(candidate.score)
            self._save_best_candidate()


class BeamSearchStrategy(SearchStrategy):
    """
    Selects the top-N candidates based on their ranking (beam width search).
    """

    def __init__(
        self,
        output_dir: pathlib.Path,
        eval_backend: EvalBackend,
        agent: LLMEnsemble,
        orig_code: str,
        prob: Prob,
        metric: str,
        simulator: str,
        give_score_feedback: float,
        give_util_feedback: float,
        give_hw_feedback: float,
        include_ancestors: bool,
        plan_icl_examples: bool,
        code_icl_examples: bool,
        num_analyses: int,
        num_plan_candidates: int,
        num_code_candidates: int,
        beam_size: int,
        num_pairs_to_combine: int,
        num_gen_per_combine: int,
        dropout_menu_options: float,
        trigger_exhaustive_threshold: float,
        trigger_exhaustive_iters: int,
        start_exhaustive_iters: int,
        prevent_duplicate_level: int,
        reimplement_failed: bool,
        translate_iters: int,
        translate_perf_threshold: float,
        translate_drop_original: bool,
        translate_score: bool,
        code_agent: LLMEnsemble = None,
        early_stop_iters: int = 0,
        early_stop_threshold: float = 1.0,
        continue_from: str | pathlib.Path | None = None,
        use_edits: bool = False,
        skip_planning: bool = False,
    ):
        self.num_analyses = num_analyses
        self.num_plan_candidates = num_plan_candidates
        self.num_code_candidates = num_code_candidates
        self.num_pairs_to_combine = num_pairs_to_combine
        self.num_gen_per_combine = num_gen_per_combine
        self.beam_size = beam_size
        self.trigger_exhaustive_threshold = trigger_exhaustive_threshold
        self.trigger_exhaustive_iters = trigger_exhaustive_iters
        self.start_exhaustive_iters = start_exhaustive_iters
        self.reimplement_failed = reimplement_failed
        self.skip_planning = skip_planning
        super().__init__(
            output_dir,
            eval_backend,
            agent,
            orig_code,
            prob,
            metric,
            simulator,
            give_score_feedback,
            give_util_feedback,
            give_hw_feedback,
            include_ancestors,
            plan_icl_examples,
            code_icl_examples,
            dropout_menu_options,
            prevent_duplicate_level,
            translate_iters,
            translate_perf_threshold,
            translate_drop_original,
            translate_score,
            code_agent=code_agent,
            early_stop_iters=early_stop_iters,
            early_stop_threshold=early_stop_threshold,
            continue_from=continue_from,
            use_edits=use_edits,
        )
        self.init_wandb()

    def filter_opt_candidates(self, opt_candidates: list) -> list:
        """
        Filter and return the top N optimization candidates based on their score.
        """
        opt_candidates.sort(key=lambda c: c.score, reverse=True)
        return opt_candidates[: self.num_opts]

    def select_candidates(self, candidates: list, num_select: int) -> list:
        return candidates[:num_select]  # Select the top-N candidates

    def propose_optimizations_iter(
        self,
        parent_candidates: list[CodeCandidate],
        save_dir: pathlib.Path,
        cur_iter: int,
        num_iters: int,
        exhaustive: bool = False,
        translate: bool = False,
    ) -> list[CodeCandidate]:
        """
        Propose a plan for each optimization in the menu.
        Returns a list of plan strings, one for each optimization in the menu.
        """
        prompt_end = (
            f"Remember that this is phase {cur_iter} out of {num_iters} optimization phases."
            if not translate
            else f"Remember that this is translation phase {cur_iter} out of {self.translate_iters} translation phases."
        )

        save_strs = []
        for parent_i, parent_candidate in enumerate(parent_candidates):
            save_str = f"parent{parent_i}"
            save_strs.append(save_str)
        kwargs = {
            "candidate_lst": parent_candidates,
            "num_plans": self.num_plan_candidates,
            "save_dir": save_dir,
            "save_strs": save_strs,
            "prob": self.prob,
            "prompt_end": prompt_end,
            "shuffle_opts": False,
            "give_score_feedback": self.give_score_feedback,
            "give_util_feedback": self.give_util_feedback,
            "give_hw_feedback": self.give_hw_feedback,
            "include_ancestors": self.include_ancestors,
            "plan_icl_examples": self.plan_icl_examples,
            "cur_iter": cur_iter,
            "num_iters": num_iters,
            "dropout_menu_options": self.dropout_menu_options,
            "translate": translate,
        }
        if exhaustive:
            # A bit of a hack: make a list pointing many times to each parent candidate so we can
            # use propose_optimizations_parallel to parallelize requests for different menu options
            kwargs["num_plans"] = len(
                self.agent.llms
            )  # Exhaustive on each LLM in the ensemble
            menu_options_lst = list(
                range(1, len(self.agent.get_opt_menu_options()) + 1)
            )
            duplicated_parent_candidates: list[CodeCandidate] = []
            duplicated_save_strs: list[str] = []
            duplicated_force_opt_menu: list[int] = []
            for parent_cand in parent_candidates:
                for menu_opt in menu_options_lst:
                    duplicated_parent_candidates.append(parent_cand)
                    duplicated_save_strs.append(save_strs[parent_cand])
                    duplicated_force_opt_menu.append(menu_opt)
            kwargs["candidate_lst"] = duplicated_parent_candidates
            kwargs["save_strs"] = duplicated_save_strs
            kwargs["force_opt_menu_lst"] = duplicated_force_opt_menu

        plan_only_candidates = self.agent.propose_optimizations_parallel(**kwargs)
        return plan_only_candidates

    def combine_parents(
        self,
        parent_candidates: list[CodeCandidate],
        num_pairs: int,
        num_to_gen: int,
        save_dir: pathlib.Path,
    ) -> list[CodeCandidate]:
        """
        Combine the code of each pair of parent candidates.
        """
        # Choose N pairs of parents to combine
        pairs = set()
        while len(pairs) < num_pairs and (
            len(pairs) < (len(parent_candidates) * (len(parent_candidates) - 1) // 2)
        ):
            i, j = random.sample(range(len(parent_candidates)), 2)
            if (i, j) not in pairs and (j, i) not in pairs:
                pairs.add((i, j))
        logger.debug("Randomly chosen pairs to combine: %s", pairs)
        # Actually generate the combined code
        combined_candidates = []
        for i, j in pairs:
            parent_i = parent_candidates[i]
            parent_j = parent_candidates[j]
            this_pair_combined_candidates = self.code_agent.combine_candidates(
                [parent_i, parent_j],
                num_to_gen,
                save_dir,
                save_str=f"{i}_{j}",
                prob=self.prob,
            )
            combined_candidates.extend(this_pair_combined_candidates)
        return combined_candidates

    def _save_run_metrics(self, all_iteration_metrics):
        run_total_s = round(
            sum(im.get("iteration_total_s", 0) for im in all_iteration_metrics), 3
        )
        run_metrics = {
            "run_total_s": run_total_s,
            "iterations": all_iteration_metrics,
        }
        try:
            total_input_tokens = 0
            total_output_tokens = 0
            total_llm_wall_s = 0.0
            total_eval_duration = 0.0
            for im in all_iteration_metrics:
                for phase in ("plan_generation", "code_generation", "context_selection", "menu_generation"):
                    for model_data in im.get(phase, {}).values():
                        total_input_tokens += model_data.get("input_tokens", 0)
                        total_output_tokens += model_data.get("output_tokens", 0)
                total_llm_wall_s += im.get("plan_duration_s", 0)
                total_llm_wall_s += im.get("code_duration_s", 0)
                total_eval_duration += im.get("evaluation", {}).get("duration_s", 0)
            run_metrics["total_input_tokens"] = total_input_tokens
            run_metrics["total_output_tokens"] = total_output_tokens
            run_metrics["total_llm_duration_s"] = round(total_llm_wall_s, 3)
            run_metrics["total_eval_duration_s"] = round(total_eval_duration, 3)
        except Exception as e:
            logger.warning("Failed to aggregate run metrics: %s", e)
        best = self._get_best_candidate()
        if best is not None:
            run_metrics["best_score"] = best.score
        try:
            with open(self.output_dir / "run_metrics.json", "w") as f:
                json.dump(run_metrics, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save run metrics: %s", e)
        total_in = run_metrics.get("total_input_tokens", 0)
        total_out = run_metrics.get("total_output_tokens", 0)
        total_tok = total_in + total_out
        def _fmt_tokens(n):
            if n >= 1_000_000:
                return f"{n / 1_000_000:.1f}M"
            if n >= 1_000:
                return f"{n / 1_000:.1f}K"
            return str(n)
        logger.info(
            "Token usage (cumulative) — input: %s, output: %s, total: %s | LLM time: %ss, eval time: %ss, total time: %ss",
            _fmt_tokens(total_in),
            _fmt_tokens(total_out),
            _fmt_tokens(total_tok),
            run_metrics.get("total_llm_duration_s", "?"),
            run_metrics.get("total_eval_duration_s", "?"),
            run_total_s,
        )

    def _save_iter_metrics_incremental(self, iter_metrics, iteration, all_iteration_metrics):
        """Save current iteration metrics to disk and update run-level aggregate."""
        try:
            all_usage = list(self.agent._usage_accumulator)
            if self.code_agent is not self.agent:
                all_usage.extend(self.code_agent._usage_accumulator)
            usage_by_phase = aggregate_usage(all_usage)
            for phase_name, model_data in usage_by_phase.items():
                iter_metrics[phase_name] = model_data
        except Exception:
            pass
        iter_metrics["iteration"] = iteration
        iter_metrics["iteration_total_s"] = round(time.perf_counter() - iter_metrics.get("_iter_t0", 0), 3)
        try:
            metrics_path = self.output_dir / f"metrics-iter-{iteration}.json"
            snapshot = {k: v for k, v in iter_metrics.items() if not k.startswith("_")}
            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    prev = json.load(f)
                for k, v in prev.items():
                    snapshot.setdefault(k, v)
            with open(metrics_path, "w") as f:
                json.dump(snapshot, f, indent=2)
        except Exception:
            pass
        current_metrics = all_iteration_metrics + [snapshot]
        self._save_run_metrics(current_metrics)

    def optimize(self, iterations: int):
        """Run the optimization process with the selected search strategy for multiple iterations."""
        losses = []
        all_iteration_metrics = []
        run_t0 = time.perf_counter()
        for i in range(1, iterations + 1):
            iter_t0 = time.perf_counter()
            iter_metrics = {"_iter_t0": iter_t0}
            cur_word = "translation" if i <= self.translate_iters else "optimization"
            logger.info(f"Iteration {i} of {cur_word}:")

            # Get current candidates (use initial code if it's the first iteration)
            cur_cand_idx = i - 1
            current_candidates = self.repository.get_candidates(cur_cand_idx)
            while len(current_candidates) == 0:
                logger.warning(
                    "No candidates found for iteration %d. Trying candidates from iteration %d.",
                    cur_cand_idx,
                    cur_cand_idx - 1,
                )
                cur_cand_idx -= 1
                current_candidates = self.repository.get_candidates(cur_cand_idx)

            cur_cand_scores = [cand.score for cand in current_candidates]
            best_loss = min(cur_cand_scores)
            wandb.log(
                {
                    f"optimize-beam-{self.prob.prob_type}-{self.prob.prob_id}-{self.simulator}": {
                        "best-loss": best_loss,
                    }
                }
            )
            losses.append(best_loss)

            if self.should_early_stop(losses, i):
                break

            # If candidates already exist for this iteration, load them and skip all other steps
            save_dir = self.output_dir / f"candidates-iter-{i}"
            num_cands_loaded = self.repository.load_candidates(i, save_dir)
            if num_cands_loaded > 0:
                logger.info("Loaded %d candidates from %s", num_cands_loaded, save_dir)
                metrics_path = self.output_dir / f"metrics-iter-{i}.json"
                if metrics_path.exists():
                    try:
                        with open(metrics_path, "r") as f:
                            all_iteration_metrics.append(json.load(f))
                    except Exception:
                        pass
                # Replay translation early-stop from cached candidates
                self._check_translation_complete(self.repository.get_candidates(i), i)
                continue

            # Step 1 + 2: Generate implementations (plan-then-implement or direct)
            translate = i <= self.translate_iters
            self.agent.reset_usage()
            if self.code_agent is not self.agent:
                self.code_agent.reset_usage()

            if self.skip_planning:
                # Direct implementation: skip planning, generate code in one shot
                save_dir = self.output_dir / f"generated-code-iter-{i}"
                save_dir.mkdir(parents=True, exist_ok=True)
                save_strs = [f"parent{p_i}" for p_i in range(len(current_candidates))]
                num_direct_samples = self.num_plan_candidates * self.num_code_candidates
                code_t0 = time.perf_counter()
                if self.use_edits:
                    impl_candidates = self.code_agent.direct_implement_code_edits_parallel(
                        current_candidates,
                        num_direct_samples,
                        save_dir,
                        save_strs=save_strs,
                        prob=self.prob,
                        give_score_feedback=self.give_score_feedback,
                        give_hw_feedback=self.give_hw_feedback,
                        include_ancestors=self.include_ancestors,
                        dropout_menu_options=self.dropout_menu_options,
                        cur_iter=i,
                        num_iters=iterations,
                        translate=translate,
                    )
                else:
                    impl_candidates = self.code_agent.direct_implement_code_parallel(
                        current_candidates,
                        num_direct_samples,
                        save_dir,
                        save_strs=save_strs,
                        prob=self.prob,
                        give_score_feedback=self.give_score_feedback,
                        give_hw_feedback=self.give_hw_feedback,
                        include_ancestors=self.include_ancestors,
                        dropout_menu_options=self.dropout_menu_options,
                        cur_iter=i,
                        num_iters=iterations,
                        translate=translate,
                    )
                code_duration = round(time.perf_counter() - code_t0, 3)
                iter_metrics["plan_duration_s"] = 0
                iter_metrics["code_duration_s"] = code_duration
                logger.info(f"Generated {len(impl_candidates)} direct implementations (no planning phase).")
                self._save_iter_metrics_incremental(iter_metrics, i, all_iteration_metrics)
            else:
                # Standard 2-phase: plan then implement
                save_dir = self.output_dir / f"generated-plans-iter-{i}"
                save_dir.mkdir(parents=True, exist_ok=True)
                exhaustive = False
                if len(losses) >= self.trigger_exhaustive_iters + 1:
                    if (
                        losses[i - 1] / losses[i - self.trigger_exhaustive_iters - 1]
                    ) >= self.trigger_exhaustive_threshold:
                        exhaustive = True
                exhaustive = exhaustive or (i <= self.start_exhaustive_iters)
                plan_t0 = time.perf_counter()
                plan_only_candidates = self.propose_optimizations_iter(
                    current_candidates,
                    save_dir,
                    i,
                    iterations,
                    exhaustive=exhaustive,
                    translate=translate,
                )
                plan_duration = round(time.perf_counter() - plan_t0, 3)
                iter_metrics["plan_duration_s"] = plan_duration
                logger.info(f"Proposed {len(plan_only_candidates)} new {cur_word} plans.")
                self._save_iter_metrics_incremental(iter_metrics, i, all_iteration_metrics)

                save_dir = self.output_dir / f"generated-code-iter-{i}"
                save_dir.mkdir(parents=True, exist_ok=True)
                save_strs = []
                for cand_idx, cand in enumerate(plan_only_candidates):
                    parent_idx = current_candidates.index(cand.parent)
                    save_strs.append(f"{parent_idx}_{cand_idx}")
                code_t0 = time.perf_counter()
                if self.use_edits:
                    impl_candidates = self.code_agent.implement_code_edits_parallel(
                        plan_only_candidates,
                        self.num_code_candidates,
                        save_dir,
                        save_strs=save_strs,
                        code_icl_examples=self.code_icl_examples,
                        prob=self.prob,
                    )
                else:
                    impl_candidates = self.code_agent.implement_code_parallel(
                        plan_only_candidates,
                        self.num_code_candidates,
                        save_dir,
                        save_strs=save_strs,
                        code_icl_examples=self.code_icl_examples,
                        prob=self.prob,
                    )
                logger.info(f"Generated {len(impl_candidates)} implementations from {len(plan_only_candidates)} plans.")
                code_duration = round(time.perf_counter() - code_t0, 3)
                iter_metrics["code_duration_s"] = code_duration
                self._save_iter_metrics_incremental(iter_metrics, i, all_iteration_metrics)

            if (
                len(current_candidates) > 1
                and self.num_pairs_to_combine > 0
                and self.num_gen_per_combine > 0
            ):
                # Try combining parents
                save_dir = self.output_dir / f"combined-code-iter-{i}"
                save_dir.mkdir(parents=True, exist_ok=True)
                combined_candidates = self.combine_parents(
                    current_candidates,
                    self.num_pairs_to_combine,
                    self.num_gen_per_combine,
                    save_dir,
                )
                impl_candidates.extend(combined_candidates)

            # Step 3: Evaluate the generated implementations
            save_dir = self.output_dir / f"eval-results-iter-{i}"
            save_dir.mkdir(parents=True, exist_ok=True)
            eval_t0 = time.perf_counter()
            evaluated_code_candidates = self.evaluate_candidates(
                impl_candidates, metric=self.metric, cur_iter=i, save_dir=save_dir
            )
            eval_duration = round(time.perf_counter() - eval_t0, 3)
            iter_metrics["evaluation"] = {
                "duration_s": eval_duration,
                "num_candidates": len(impl_candidates),
            }
            logger.info(f"Evaluated {len(evaluated_code_candidates)} implementations.")
            self._save_iter_metrics_incremental(iter_metrics, i, all_iteration_metrics)

            # Step 3.5: Reimplement failed implementations
            if self.reimplement_failed:
                failed_candidates = [
                    c
                    for c in evaluated_code_candidates
                    if c.score == float("inf") and (c.stdout or c.stderr)
                ]
                if len(failed_candidates) > 0:
                    logger.info(
                        f"Found {len(failed_candidates)} failed implementations with error output. Attempting to reimplement..."
                    )
                    save_dir = self.output_dir / f"reimplemented-code-iter-{i}"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_strs = [
                        f"failed_{idx}" for idx in range(len(failed_candidates))
                    ]
                    if self.use_edits:
                        reimplemented_candidates = (
                            self.code_agent.reimplement_failed_code_edits_parallel(
                                failed_candidates,
                                1,
                                save_dir,
                                save_strs=save_strs,
                                prob=self.prob,
                            )
                        )
                    else:
                        reimplemented_candidates = (
                            self.code_agent.reimplement_failed_code_parallel(
                                failed_candidates,
                                1,
                                save_dir,
                                save_strs=save_strs,
                                prob=self.prob,
                            )
                        )
                    logger.info(
                        f"Generated {len(reimplemented_candidates)} reimplementations."
                    )

                    # Evaluate the reimplementations
                    save_dir = self.output_dir / f"eval-reimplemented-results-iter-{i}"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    reimplemented_evaluated = self.evaluate_candidates(
                        reimplemented_candidates,
                        metric=self.metric,
                        cur_iter=i,
                        save_dir=save_dir,
                    )
                    logger.info(
                        f"Evaluated {len(reimplemented_evaluated)} reimplementations."
                    )

                    # Add successful reimplementations to the evaluated list
                    evaluated_code_candidates.extend(reimplemented_evaluated)

            # Step 3.75: Score translation completeness during translate iters
            if translate and self.translate_score:
                correct_candidates = [
                    c for c in evaluated_code_candidates if c.score != float("inf")
                ]
                if correct_candidates:
                    ts_save_dir = self.output_dir / f"eval-results-iter-{i}"
                    self._score_translation(correct_candidates, ts_save_dir)
                    logger.info(
                        "Translation scores: %s",
                        [
                            (f"{c.translation_score:.1f}", f"{c.score:.4f}")
                            for c in correct_candidates
                        ],
                    )

            # Step 4: Filter and rank the implementations
            improving_candidates = self.filter_code_candidates(
                evaluated_code_candidates, cur_iter=i, num_iters=iterations
            )
            cands_to_filter = improving_candidates + current_candidates
            candidates_for_next_iter = self.filter_code_candidates(
                cands_to_filter,
                num_to_keep=self.beam_size,
                cur_iter=i,
                num_iters=iterations,
            )

            # Early-stop translation only when every beam candidate is fully translated
            if translate:
                self._check_translation_complete(candidates_for_next_iter, i)

            # On the final translation iteration (natural or early-stopped),
            # re-select the beam excluding the original so all slots go to translated candidates.
            if self.translate_drop_original and i == self.translate_iters:
                cands_no_original = [c for c in cands_to_filter if c.parent is not None]
                candidates_for_next_iter = self.filter_code_candidates(
                    cands_no_original,
                    num_to_keep=self.beam_size,
                    cur_iter=i,
                    num_iters=iterations,
                )

            candidates_for_next_iter = self.add_feedback(candidates_for_next_iter)

            # Step 5: Save the improving candidates and update the repository
            self.repository.add_candidates(improving_candidates, "improving")
            self.repository.add_candidates(candidates_for_next_iter, i)
            logger.info(
                f"Filtered down to {len(candidates_for_next_iter)} code candidates."
            )
            logger.info(
                f"Saved {len(improving_candidates)} improving code candidates to repository."
            )

            # Step 6: Save the latest candidates to disk
            save_dir = self.output_dir / f"candidates-iter-{i}"
            save_dir.mkdir(parents=True, exist_ok=True)
            self.repository.save_candidates(i, save_dir)
            # Show latest candidates
            logger.info("New candidate scores:")
            for candidate in candidates_for_next_iter:
                logger.info(candidate.score)
            self._save_best_candidate()

            # Final save for this iteration (captures complete usage data)
            self._save_iter_metrics_incremental(iter_metrics, i, all_iteration_metrics)
            final_snapshot = {k: v for k, v in iter_metrics.items() if not k.startswith("_")}
            all_iteration_metrics.append(final_snapshot)

        self._save_run_metrics(all_iteration_metrics)
        best = self._get_best_candidate()
        initial_candidates = self.repository.get_candidates(0)
        initial_score = initial_candidates[0].score if initial_candidates and initial_candidates[0].score is not None else None
        elapsed = time.perf_counter() - run_t0

        if best and best.score is not None:
            wandb.log(
                {
                    f"optimize-beam-{self.prob.prob_type}-{self.prob.prob_id}-{self.simulator}": {
                        "best-loss": best.score,
                    }
                }
            )

        logger.info("=" * 60)
        logger.info("Optimization complete. %d iterations in %.1f minutes.", len(all_iteration_metrics), elapsed / 60)
        if initial_score is not None:
            logger.info("Initial score: %.3f", initial_score)
        if best and best.score is not None:
            logger.info("Best score: %.3f", best.score)
            if initial_score and initial_score > 0:
                logger.info("Speedup: %.2fx", initial_score / best.score)
        logger.info("=" * 60)
        wandb.finish()

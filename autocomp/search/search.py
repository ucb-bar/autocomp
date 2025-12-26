import pathlib
import random
import json

import wandb

from autocomp.common import logger
from autocomp.search.code_repo import CodeCandidate, CodeRepository
from autocomp.search.llm_agent import GemminiLLMAgent, CudaLLMAgent, TrnLLMAgent
from autocomp.search.llm_ensemble import LLMEnsemble
from autocomp.backend.hardware_backend import HardwareBackend
from autocomp.backend.gemmini_eval import GemminiHardwareBackend
from autocomp.backend.kb_eval import KBHardwareBackend, KERNELBENCH_DIR
from autocomp.backend.trn_eval import TrnHardwareBackend
from autocomp.search.prob import Prob

class SearchStrategy:
    """
    Base class for different search strategies.
    """
    def __init__(self, 
                 output_dir: pathlib.Path,
                 hw_backend: HardwareBackend,
                 llm: LLMEnsemble,
                 orig_code: str,
                 prob: Prob,
                 metric: str,
                 simulator: str,
                 give_score_feedback: float,
                 give_util_feedback: float,
                 give_spad_acc_feedback: float,
                 include_ancestors: bool,
                 plan_icl_examples: bool,
                 code_icl_examples: bool,
                 dropout_menu_options: float,
                 prevent_duplicate_level: int,
                 translate_iters: int,
                 translate_perf_threshold: float,
                 code_llm: LLMEnsemble = None,
               ):
        self.repository = CodeRepository()  # Stores the code candidates
        self.llm = llm  # The LLM used to propose optimizations (planning)
        self.code_llm = code_llm if code_llm is not None else llm  # The LLM used for code implementation
        self.prob = prob
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hw_backend = hw_backend
        self.metric = metric
        self.simulator = simulator
        self.give_score_feedback = give_score_feedback
        self.give_util_feedback = give_util_feedback
        self.give_spad_acc_feedback = give_spad_acc_feedback
        self.include_ancestors = include_ancestors
        self.plan_icl_examples = plan_icl_examples
        self.code_icl_examples = code_icl_examples
        self.dropout_menu_options = dropout_menu_options
        self.prevent_duplicate_level = prevent_duplicate_level
        self.translate_iters = translate_iters
        self.translate_perf_threshold = translate_perf_threshold
        save_dir = self.output_dir / f"candidates-iter-0"
        save_dir.mkdir(parents=True, exist_ok=True)
        num_cands_loaded = self.repository.load_candidates(0, save_dir)
        if num_cands_loaded > 0:
            logger.info("Loaded initial code from %s", save_dir)
        else:
            orig_code_candidate = CodeCandidate(None, None, orig_code)
            self.evaluate_candidates([orig_code_candidate], self.metric) # Evaluate the initial code
            if orig_code_candidate.score == float("inf"):
                raise ValueError("Initial code is incorrect.")
            self.add_feedback([orig_code_candidate])
            self.repository.add_candidates([orig_code_candidate], 0)  # Add the initial code as the first candidate
            self.repository.save_candidates(0, save_dir)
        initial_code_candidates: list[CodeCandidate] = self.repository.get_candidates(0)
        logger.info("Initial code scores:")
        for candidate in initial_code_candidates:
            logger.info(candidate.score)

    def propose_optimizations_iter(self, candidates: list[CodeCandidate], num_plans: int) -> list[CodeCandidate]:
        """
        Use the LLM to propose new optimization plans for the given code.
        """
        raise NotImplementedError
    
    def evaluate_candidates(self, candidates: list[CodeCandidate], metric: str, cur_iter: int=None, save_dir: pathlib.Path=None) -> list[CodeCandidate]:
        """
        Evaluate the candidates based on the provided optimization metric
        and update their scores.
        """
        # Load stats if they already exist in the save_dir
        if save_dir is not None and save_dir.exists() and all((save_dir / f"code_{i}_result.txt").exists() for i in range(len(candidates))):
            logger.info(f"Loading cached evaluation results for all {len(candidates)} candidates from {save_dir}")
            per_cand_stats = []
            for i in range(len(candidates)):
                with open(save_dir / f"code_{i}_result.txt", "r") as f:
                    per_cand_stats.append(json.load(f))
        else:
            per_cand_stats = self.hw_backend.evaluate_code(self.prob, [candidate.code for candidate in candidates], self.simulator)

            # Save stats
            if save_dir is not None:
                for cand_i, stats in enumerate(per_cand_stats):
                    with open(save_dir / f"code_{cand_i}_result.txt", "w") as f:
                        json.dump(stats, f, indent=4)
                    with open(save_dir / f"code_{cand_i}_result_full.txt", "w") as f:
                        f.write(str(stats).replace("\\n", "\n"))
                        if candidates[cand_i].parent is not None:
                            f.write("\nPrev latency: "+str(candidates[cand_i].parent.score)+"\n")
                        if stats["correct"]:
                            f.write("New latency: "+ str(stats[metric]) +"\n")
                        else:
                            f.write("New latency: N/A\n")
                        f.write("Plan: " + candidates[cand_i].plan.replace("\\n", "\n") +"\n")
                        f.write("\n"+repr(candidates[cand_i]))

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

    def add_feedback(self, candidates: list[CodeCandidate]) -> list[CodeCandidate]:
        # NOTE: Assumes that feedback for a particular candidate only gets added once and does not get updated
        for cand_i in range(len(candidates)):
            if candidates[cand_i].spad_acc_stats: # If already has feedback, skip
                continue
            if self.give_spad_acc_feedback > 0:
                spad_acc_stats = self.hw_backend.get_spad_acc_utilization(self.prob, [candidates[cand_i].code])[0]
                spad_size_kb = self.hw_backend.spad_size_kb
                acc_size_kb = self.hw_backend.acc_size_kb
                spad_cap_used = round(spad_acc_stats['spad_util'] * spad_size_kb)
                acc_cap_used = round(spad_acc_stats['acc_util'] * acc_size_kb)
                feedback = [
                    f"Scratchpad utilization is {spad_cap_used}KB out of {spad_size_kb}KB.",
                    f"Accumulator utilization is {acc_cap_used}KB out of {acc_size_kb}KB."
                ]
                if spad_acc_stats['spad_util'] < 1:
                    feedback[0] += " Consider increasing scratchpad utilization to improve performance."
                if spad_acc_stats['acc_util'] < 1:
                    feedback[1] += " Consider increasing accumulator utilization to improve performance."
                logger.debug("Adding feedback to candidate %d: %s", cand_i, feedback)
                candidates[cand_i].update_spad_acc_stats(feedback)
        return candidates

    def filter_code_candidates(self, code_candidates: list[CodeCandidate], num_to_keep: int | None = None, cur_iter: int = None, num_iters: int = None) -> list:
        """
        Filter and return the top N code candidates based on their score.
        If N (num_to_keep) is not provided, will return all candidates with a higher score than their parent.
        """
        # Filter out incorrect candidates
        code_candidates = [c for c in code_candidates if c.score != float("inf")]

        keep_factor = self.translate_perf_threshold if self.translate_iters >= cur_iter else 1
        # if cur_iter is not None and num_iters is not None:
        #     if cur_iter <= 2:
        #         keep_factor = 1.5
        if num_to_keep is None:
            cur_candidates = []
            for c in code_candidates:
                if isinstance(c.parent, list):
                    if all([c.score < p.score for p in c.parent]):
                        cur_candidates.append(c)
                else:
                    if c.score < c.parent.score * keep_factor:
                        if c.score >= c.parent.score:
                            logger.debug(f"Keep factor used at iter {cur_iter}, candidate\n{c}")
                        cur_candidates.append(c)
        else:
            cur_candidates = []
            code_candidates.sort(key=lambda c: c.score, reverse=False)
            for cand in code_candidates:
                if len(cur_candidates) >= num_to_keep:
                    break
                if cand.parent is None:
                    cur_candidates.append(cand)
                    continue
                # Don't keep any candidates with a score same or higher than their parent
                if cand.score >= cand.parent.score * keep_factor:
                    logger.debug(f"Working candidate has score {cand.score} >= parent score {cand.parent.score}.")
                    logger.debug(f"Candidate plan:\n{cand.plan}")
                    logger.debug(f"Candidate code:\n{cand.code}")
                    continue
                dont_add = False
                if self.prevent_duplicate_level == 2:
                    # Don't keep any candidates with parents in common other than the root candidate
                    # And also don't keep the parents of candidates already in the list
                    for already_in_cand in cur_candidates:
                        if dont_add == True:
                            break
                        if (cand.parent == already_in_cand.parent) and (cand.plan == already_in_cand.plan):
                            dont_add = True
                            break
                        already_in_parent_cand = already_in_cand.parent
                        while already_in_parent_cand is not None:
                            if (cand==already_in_parent_cand or cand.parent == already_in_parent_cand) and (already_in_parent_cand.parent is not None):
                                dont_add = True
                                break
                            already_in_parent_cand = already_in_parent_cand.parent
                elif self.prevent_duplicate_level == 1:
                    # Don't keep any candidates with same parent as one already in the list, unless that parent is the root candidate
                    for already_in_cand in cur_candidates:
                        if cand.parent == already_in_cand.parent and cand.parent.parent is not None:
                            dont_add = True
                else:
                    # Don't keep any candidates with same parent and plan as one already in the list (the one already in the list is equal or better)
                    for already_in_cand in cur_candidates:
                        if (cand.parent == already_in_cand.parent) and (cand.plan == already_in_cand.plan):
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

class ExhaustiveSearchStrategy(SearchStrategy):
    """
    Manages the iterative optimization process, including proposing, filtering, 
    and refining code candidates.
    """
    def propose_optimizations_iter(self, parent_candidates: list[CodeCandidate], save_dir: pathlib.Path) -> list[CodeCandidate]:
        """
        Propose a plan for each optimization in the menu.
        Returns a list of plan strings, one for each optimization in the menu.
        """
        unimplemented_candidates = []
        for candidate in parent_candidates:
            for opt_menu_option in range(len(self.llm.get_opt_menu_options())):
                new_cand = self.llm.propose_optimizations(candidate, num_plans=1, save_dir=save_dir, save_str=None, prob=self.prob, force_opt_menu=opt_menu_option)[0]
                unimplemented_candidates.append(new_cand)
        return unimplemented_candidates

    def optimize(self, iterations: int):
        """Run the optimization process with the selected search strategy for multiple iterations."""
        for i in range(1, iterations + 1):
            logger.info(f"Iteration {i} of optimization:")

            # Get current candidates (use initial code if it's the first iteration)
            cur_cand_idx = i-1
            current_candidates = self.repository.get_candidates(cur_cand_idx)
            while len(current_candidates) == 0:
                logger.warning("No candidates found for iteration %d. Trying candidates from iteration %d.", cur_cand_idx, cur_cand_idx-1)
                cur_cand_idx -= 1
                current_candidates = self.repository.get_candidates(cur_cand_idx)

            # Step 1: Propose optimizations for each candidate
            save_dir = self.output_dir / f"generated-plans-iter-{i}"
            save_dir.mkdir(parents=True, exist_ok=True)
            plan_only_candidates = self.propose_optimizations_iter(current_candidates, save_dir)
            logger.info(f"Proposed {len(plan_only_candidates)} new optimizations.")

            # Step 2: Generate code candidates for each optimization plan
            save_dir = self.output_dir / f"generated-code-iter-{i}"
            save_dir.mkdir(parents=True, exist_ok=True)
            impl_candidates = []
            for plan_idx, plan_only_cand in enumerate(plan_only_candidates):
                parent_idx = current_candidates.index(plan_only_cand.parent)
                this_plan_impl_candidates = self.code_llm.implement_code(plan_only_cand, 1, save_dir, save_str=f"{parent_idx}_{plan_idx}", prob=self.prob)
                impl_candidates.extend(this_plan_impl_candidates)
            logger.info(f"Generated {len(impl_candidates)} implementations.")

            # Step 3: Evaluate the code candidates
            save_dir = self.output_dir / f"eval-results-iter-{i}"
            save_dir.mkdir(parents=True, exist_ok=True)
            evaluated_code_candidates = self.evaluate_candidates(impl_candidates, metric=self.metric, cur_iter=i, save_dir=save_dir)
            logger.info(f"Evaluated {len(evaluated_code_candidates)} code candidates.")

            # Step 4: Filter and rank the code candidates
            improving_candidates = self.filter_code_candidates(evaluated_code_candidates) # No num_to_keep means keep all improving candidates
            candidates_for_next_iter = self.filter_code_candidates(improving_candidates, num_to_keep=1)

            # Step 5: Save the improving candidates and update the repository
            self.repository.add_candidates(improving_candidates, "improving")
            self.repository.add_candidates(candidates_for_next_iter, i)
            logger.info(f"Filtered down to {len(candidates_for_next_iter)} code candidates.")
            logger.info(f"Saved {len(improving_candidates)} improving code candidates to repository.")

            # Step 6: Save the latest candidates to disk
            save_dir = self.output_dir / f"candidates-iter-{i}"
            save_dir.mkdir(parents=True, exist_ok=True)
            self.repository.save_candidates(i, save_dir)
            # Show latest candidates
            logger.info("New candidate scores:")
            for candidate in candidates_for_next_iter:
                logger.info(candidate.score)

class BeamSearchStrategy(SearchStrategy):
    """
    Selects the top-N candidates based on their ranking (beam width search).
    """
    def __init__(self,
                 output_dir: pathlib.Path,
                 backend: str,
                 llm: LLMEnsemble,
                 orig_code: str,
                 prob: Prob,
                 metric: str,
                 simulator: str,
                 give_score_feedback: float,
                 give_util_feedback: float,
                 give_spad_acc_feedback: float,
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
                 code_llm: LLMEnsemble = None,
                ):
        super().__init__(output_dir, backend, llm, orig_code, prob, metric, simulator, give_score_feedback, give_util_feedback, give_spad_acc_feedback, include_ancestors, plan_icl_examples, code_icl_examples, dropout_menu_options, prevent_duplicate_level, translate_iters, translate_perf_threshold, code_llm=code_llm)
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
        self.init_wandb()

    def filter_opt_candidates(self, opt_candidates: list) -> list:
        """
        Filter and return the top N optimization candidates based on their score.
        """
        opt_candidates.sort(key=lambda c: c.score, reverse=True)
        return opt_candidates[:self.num_opts]

    def select_candidates(self, candidates: list, num_select: int) -> list:
        return candidates[:num_select]  # Select the top-N candidates

    def propose_optimizations_iter(self, parent_candidates: list[CodeCandidate], save_dir: pathlib.Path, cur_iter: int, num_iters: int,
                                   exhaustive: bool = False, translate: bool = False) -> list[CodeCandidate]:
        """
        Propose a plan for each optimization in the menu.
        Returns a list of plan strings, one for each optimization in the menu.
        """
        prompt_end = f"Remember that this is phase {cur_iter} out of {num_iters} optimization phases."
        # if cur_iter <= num_iters // 2:
        #     prompt_end += " In early phases of optimization, we suggest optimizing loop tiling (1) or loop ordering (2)."
        # prompt_end += " Try to use an optimization that has not been used in prior phases."

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
            "give_spad_acc_feedback": self.give_spad_acc_feedback,
            "include_ancestors": self.include_ancestors,
            "plan_icl_examples": self.plan_icl_examples,
            "cur_iter": cur_iter,
            "num_iters": num_iters,
            "dropout_menu_options": self.dropout_menu_options,
            "translate": translate,
        }
        # kwargs["force_opt_menu"] = 3
        # new_cands = self.llm.propose_optimizations(parent_candidate, **kwargs)
        # unimplemented_candidates.extend(new_cands)
        if exhaustive:
            # A bit of a hack: make a list pointing many times to each parent candidate so we can 
            # use propose_optimizations_parallel to parallelize requests for different menu options
            kwargs["num_plans"] = len(self.llm.llms) # Exhaustive on each LLM in the ensemble
            menu_options_lst = list(range(1, len(self.llm.get_opt_menu_options())+1))
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
        
        new_cands = self.llm.propose_optimizations_parallel(**kwargs)
        return new_cands

    def combine_parents(self, parent_candidates: list[CodeCandidate], num_pairs: int, num_to_gen: int, save_dir: pathlib.Path) -> list[CodeCandidate]:
        """
        Combine the code of each pair of parent candidates.
        """
        # Choose N pairs of parents to combine
        pairs = set()
        while len(pairs) < num_pairs and (len(pairs) < (len(parent_candidates) * (len(parent_candidates) - 1) // 2)):
            i, j = random.sample(range(len(parent_candidates)), 2)
            if (i, j) not in pairs and (j, i) not in pairs:
                pairs.add((i, j))
        logger.debug("Randomly chosen pairs to combine: %s", pairs)
        # Actually generate the combined code
        combined_candidates = []
        for i, j in pairs:
            parent_i = parent_candidates[i]
            parent_j = parent_candidates[j]
            this_pair_combined_candidates = self.code_llm.combine_candidates([parent_i, parent_j], num_to_gen, save_dir, save_str=f"{i}_{j}")
            combined_candidates.extend(this_pair_combined_candidates)
        return combined_candidates

    def optimize(self, iterations: int):
        """Run the optimization process with the selected search strategy for multiple iterations."""
        losses = []
        for i in range(1, iterations + 1):
            logger.info(f"Iteration {i} of optimization:")

            # Get current candidates (use initial code if it's the first iteration)
            cur_cand_idx = i-1
            current_candidates = self.repository.get_candidates(cur_cand_idx)
            while len(current_candidates) == 0:
                logger.warning("No candidates found for iteration %d. Trying candidates from iteration %d.", cur_cand_idx, cur_cand_idx-1)
                cur_cand_idx -= 1
                current_candidates = self.repository.get_candidates(cur_cand_idx)

            cur_cand_scores = [cand.score for cand in current_candidates]
            best_loss = min(cur_cand_scores)
            wandb.log({f"optimize-beam-{self.prob.prob_type}-{self.prob.prob_id}-{self.simulator}": {
                "best-loss": best_loss,
            }})
            losses.append(best_loss)

            # If candidates already exist for this iteration, load them and skip all other steps
            save_dir = self.output_dir / f"candidates-iter-{i}"
            num_cands_loaded = self.repository.load_candidates(i, save_dir)
            if num_cands_loaded > 0:
                logger.info("Loaded %d candidates from %s", num_cands_loaded, save_dir)
                continue

            # Step 1: Propose optimizations for each candidate
            save_dir = self.output_dir / f"generated-plans-iter-{i}"
            save_dir.mkdir(parents=True, exist_ok=True)
            # If we don't improve by trigger_exhaustive_threshold for trigger_exhaustive_iters, do exhaustive
            exhaustive = False
            if len(losses) >= self.trigger_exhaustive_iters+1:
                if (losses[i-1] / losses[i-self.trigger_exhaustive_iters-1]) >= self.trigger_exhaustive_threshold:
                    exhaustive = True
            exhaustive = exhaustive or (i <= self.start_exhaustive_iters) # or if we are in the first start_exhaustive_iters iterations
            translate = (i <= self.translate_iters)
            plan_only_candidates = self.propose_optimizations_iter(current_candidates, save_dir, i, iterations, 
                                                                   exhaustive=exhaustive, translate=translate)
            logger.info(f"Proposed {len(plan_only_candidates)} new optimizations.")

            # Step 2: Generate code candidates for each optimization plan
            save_dir = self.output_dir / f"generated-code-iter-{i}"
            save_dir.mkdir(parents=True, exist_ok=True)
            # impl_candidates = []
            # for plan_idx, plan_only_cand in enumerate(plan_only_candidates):
            #     parent_idx = current_candidates.index(plan_only_cand.parent)
            #     this_plan_impl_candidates = self.llm.implement_code(plan_only_cand, self.num_code_candidates, save_dir, save_str=f"{parent_idx}_{plan_idx}")
            #     impl_candidates.extend(this_plan_impl_candidates)
            save_strs = []
            for cand_idx, cand in enumerate(plan_only_candidates):
                parent_idx = current_candidates.index(cand.parent)
                save_strs.append(f"{parent_idx}_{cand_idx}")
            impl_candidates = self.code_llm.implement_code_parallel(plan_only_candidates, self.num_code_candidates, save_dir, save_strs=save_strs, code_icl_examples=self.code_icl_examples, prob=self.prob)
            logger.info(f"Generated {len(impl_candidates)} implementations.")

            if len(current_candidates) > 1 and self.num_pairs_to_combine > 0 and self.num_gen_per_combine > 0:
                # Try combining parents
                save_dir = self.output_dir / f"combined-code-iter-{i}"
                save_dir.mkdir(parents=True, exist_ok=True)
                combined_candidates = self.combine_parents(current_candidates, self.num_pairs_to_combine, self.num_gen_per_combine, save_dir)
                impl_candidates.extend(combined_candidates)

            # Step 3: Evaluate the code candidates
            save_dir = self.output_dir / f"eval-results-iter-{i}"
            save_dir.mkdir(parents=True, exist_ok=True)
            evaluated_code_candidates = self.evaluate_candidates(impl_candidates, metric=self.metric, cur_iter=i, save_dir=save_dir)
            logger.info(f"Evaluated {len(evaluated_code_candidates)} code candidates.")

            # Step 3.5: Reimplement failed candidates
            if self.reimplement_failed:
                failed_candidates = [c for c in evaluated_code_candidates if c.score == float("inf") and (c.stdout or c.stderr)]
                if len(failed_candidates) > 0:
                    logger.info(f"Found {len(failed_candidates)} failed candidates with error output. Attempting to reimplement...")
                    save_dir = self.output_dir / f"reimplemented-code-iter-{i}"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_strs = [f"failed_{idx}" for idx in range(len(failed_candidates))]
                    # Reimplement with the same number of samples as original code candidates
                    reimplemented_candidates = self.code_llm.reimplement_failed_code_parallel(failed_candidates, 1, save_dir, save_strs=save_strs, prob=self.prob)
                    logger.info(f"Generated {len(reimplemented_candidates)} reimplemented candidates.")
                    
                    # Evaluate the reimplemented candidates
                    save_dir = self.output_dir / f"eval-reimplemented-results-iter-{i}"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    reimplemented_evaluated = self.evaluate_candidates(reimplemented_candidates, metric=self.metric, cur_iter=i, save_dir=save_dir)
                    logger.info(f"Evaluated {len(reimplemented_evaluated)} reimplemented candidates.")
                    
                    # Add successful reimplementations to the evaluated candidates list
                    evaluated_code_candidates.extend(reimplemented_evaluated)

            # Step 4: Filter and rank the code candidates
            improving_candidates = self.filter_code_candidates(evaluated_code_candidates, cur_iter=i, num_iters=iterations) # No num_to_keep means keep all improving candidates
            # Keep the beam_size best candidates out of the existing and new candidates
            # if i <= iterations//3:
            #     cands_to_filter = evaluated_code_candidates + current_candidates
            # else:
            #     cands_to_filter = improving_candidates + current_candidates
            cands_to_filter = improving_candidates + current_candidates
            candidates_for_next_iter = self.filter_code_candidates(cands_to_filter, num_to_keep=self.beam_size, cur_iter=i, num_iters=iterations)
            candidates_for_next_iter = self.add_feedback(candidates_for_next_iter)

            # Step 5: Save the improving candidates and update the repository
            self.repository.add_candidates(improving_candidates, "improving")
            self.repository.add_candidates(candidates_for_next_iter, i)
            logger.info(f"Filtered down to {len(candidates_for_next_iter)} code candidates.")
            logger.info(f"Saved {len(improving_candidates)} improving code candidates to repository.")

            # Step 6: Save the latest candidates to disk
            save_dir = self.output_dir / f"candidates-iter-{i}"
            save_dir.mkdir(parents=True, exist_ok=True)
            self.repository.save_candidates(i, save_dir)
            # Show latest candidates
            logger.info("New candidate scores:")
            for candidate in candidates_for_next_iter:
                logger.info(candidate.score)

        last_iter_cands = self.repository.get_candidates(iterations)
        wandb.log({f"optimize-beam-{self.prob.prob_type}-{self.prob.prob_id}-{self.simulator}": {
            "best-loss": min([cand.score for cand in last_iter_cands]),
        }})

def main():
    # Generic search parameters
    backend = "trn"  # Options: "gemmini", "trn", "cuda"
    # Models are specified as "provider::model"
    # Valid providers are "openai", "gcp", "aws", "anthropic", "mistralgcp", "together", "vllm"
    # If no provider is specified, the provider is inferred from the model name
    models = ["openai::o4-mini", "openai::gpt-5.2", "gcp::gemini-3-pro-preview", "aws::us.anthropic.claude-opus-4-5-20251101-v1:0"]  # Models for planning
    code_models = ["gcp::gemini-3-pro-preview", "openai::gpt-5.2"] # Models for code implementation (None means use same as planning models)
    metric = "latency"
    simulator = "trn" # "firesim" or "spike" if backend == "gemmini"; "kernelbench" if backend == "cuda"; "trn" if backend == "trn"
    search_strategy = "beam"
    iterations = 8
    prob_type = "trn-e2e" # see README.md or sols directory for available problems
    prob_id = 11

    # Beam search parameters
    num_plan_candidates=5
    num_code_candidates=2
    beam_size=6

    # Translation parameters
    translate_iters = 0
    translate_perf_threshold = 1.05

    # Planning prompt knobs
    dropout_menu_options = 0.2
    give_score_feedback = 1
    give_util_feedback = 0
    give_spad_acc_feedback = 0
    include_ancestors = False
    plan_icl_examples = False
    code_icl_examples = False

    # Typically not used
    num_analyses=0
    num_pairs_to_combine = 0
    num_gen_per_combine = 0
    trigger_exhaustive_threshold = 1
    trigger_exhaustive_iters = 20
    start_exhaustive_iters = 0
    random.seed(1111)
    
    # prevent_duplicate_level
    # 0: prevent candidates with same parent and plan
    # 1: prevent candidates with same parent
    # 2: prevent candidates with any parents in common (any nodes below root have branching factor 1)
    prevent_duplicate_level = 0
    
    # Reimplement failed candidates
    # Only works for trn
    reimplement_failed = True

    # Sanitize model names for file system compatibility
    for i in range(len(models)):
        models[i] = models[i].replace("/", "_")
    if code_models is not None:
        for i in range(len(code_models)):
            code_models[i] = code_models[i].replace("/", "_")

    output_str = f"{prob_type}_{prob_id}_{search_strategy}_iters{iterations}_{simulator}"
    for model in models:
        output_str += f"_{model[:15]}"
    if code_models is not None:
        output_str += "_code"
        for model in code_models:
            output_str += f"_{model[:15]}"
    output_str += f"_dropout{dropout_menu_options}"
    if search_strategy == "beam":
        output_str += f"_analyses{num_analyses}_plan{num_plan_candidates}_code{num_code_candidates}_beam{beam_size}"
    if translate_iters > 0:
        output_str += f"_translate{translate_iters}_{translate_perf_threshold}"
    output_str += f"_score{give_score_feedback}_util{give_util_feedback}_spadacc{give_spad_acc_feedback}_ancestors{int(include_ancestors)}_preventdupe{prevent_duplicate_level}_planicl{int(plan_icl_examples)}_codeicl{int(code_icl_examples)}"
    output_dir = pathlib.Path("output/" + output_str)

    output_dir.mkdir(parents=True, exist_ok=True)
    import autocomp.common.my_logging
    autocomp.common.my_logging.move_log(output_dir)

    logger.info(f"Output directory: {output_dir}")

    # Get initial code
    prob = Prob(prob_type, prob_id)
    if backend == "cuda":
        if "kb-" in prob_type:
            level_str = prob_type.split("-")[1]
            kb_level_dir = pathlib.Path(KERNELBENCH_DIR) / "KernelBench" / level_str
            matches = list(kb_level_dir.glob(f"{prob_id}_*.py"))
            if not matches:
                raise FileNotFoundError(f"No solution file found matching pattern {prob_id}_*.py in {kb_level_dir}")
            with open(matches[0]) as f:
                initial_code = f.read().replace("Model", "ModelNew")
        else:
            # Find file matching pattern
            sol_dir = pathlib.Path(__file__).parent.parent.parent / "sols" / prob_type
            matches = list(sol_dir.glob(f"{prob_id}_*.py"))
            if not matches:
                raise FileNotFoundError(f"No solution file found matching pattern {prob_id}_*.py in {sol_dir}")
            with open(matches[0]) as f:
                initial_code = f.read()
    elif backend == "gemmini":
        if "admm" in prob_type:
            with open(pathlib.Path(__file__).parent.parent.parent / "sols" / "admm-multifunction" / f"sol{prob_id}_unopt_sw.c") as f:
                initial_code = f.read()
        else:
            with open(pathlib.Path(__file__).parent.parent.parent / "sols" / prob_type / f"sol{prob_id}_exo_baseline.c") as f:
                initial_code = f.read()
    elif backend == "trn":
        # Find file matching pattern for Trainium/NKI kernels
        sol_dir = pathlib.Path(__file__).parent.parent.parent / "sols" / prob.prob_type
        matches = list(sol_dir.glob(f"{prob_id}_*.py"))
        if not matches:
            raise FileNotFoundError(f"No solution file found matching pattern {prob_id}_*.py in {sol_dir}")
        with open(matches[0]) as f:
            initial_code = f.read()

    # Initialize hardware backend and LLM ensemble
    if backend == "cuda":
        if simulator == "kernelbench":
            hw_backend = KBHardwareBackend()
        else:
            hw_backend = GpuModeHardwareBackend()
        llm = LLMEnsemble([CudaLLMAgent(model) for model in models])
        code_llm = LLMEnsemble([CudaLLMAgent(model) for model in code_models]) if code_models is not None else None
    elif backend == "gemmini":
        spad_size_kb = 256
        acc_size_kb = 64
        if "admm" in prob.prob_type:
            pe_dim = 4
        elif "exo" in prob.prob_type or "gemm" in prob.prob_type:
            pe_dim = 16
        elif "gpt" in prob.prob_type:
            pe_dim = 32
            spad_size_kb = 512
            acc_size_kb = 128
        hw_backend = GemminiHardwareBackend(pe_dim, spad_size_kb, acc_size_kb)
        llm = LLMEnsemble([GemminiLLMAgent(model, pe_dim) for model in models])
        code_llm = LLMEnsemble([GemminiLLMAgent(model, pe_dim) for model in code_models]) if code_models is not None else None
    elif backend == "trn":
        hw_backend = TrnHardwareBackend()
        llm = LLMEnsemble([TrnLLMAgent(model) for model in models])
        code_llm = LLMEnsemble([TrnLLMAgent(model) for model in code_models]) if code_models is not None else None
    else:
        raise ValueError(f"Unknown backend: {backend}")
    if search_strategy == "exhaustive":
        optimizer = ExhaustiveSearchStrategy(output_dir, hw_backend, llm, initial_code, prob, metric, simulator, give_score_feedback, give_util_feedback, give_spad_acc_feedback, include_ancestors, plan_icl_examples, code_icl_examples, dropout_menu_options, prevent_duplicate_level, code_llm=code_llm)
    elif search_strategy == "beam":
        optimizer = BeamSearchStrategy(output_dir, hw_backend, llm, initial_code, prob, metric, simulator, give_score_feedback, give_util_feedback, give_spad_acc_feedback, include_ancestors, plan_icl_examples, code_icl_examples,
                                       num_analyses=num_analyses, num_plan_candidates=num_plan_candidates, num_code_candidates=num_code_candidates, beam_size=beam_size,
                                       num_pairs_to_combine=num_pairs_to_combine, num_gen_per_combine=num_gen_per_combine, 
                                       dropout_menu_options=dropout_menu_options, trigger_exhaustive_threshold=trigger_exhaustive_threshold, trigger_exhaustive_iters=trigger_exhaustive_iters, start_exhaustive_iters=start_exhaustive_iters,
                                       prevent_duplicate_level=prevent_duplicate_level, reimplement_failed=reimplement_failed, translate_iters=translate_iters, translate_perf_threshold=translate_perf_threshold, code_llm=code_llm)

    # Start the optimization process
    optimizer.optimize(iterations)

if __name__ == "__main__":
    main()

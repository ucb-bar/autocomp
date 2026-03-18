import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from autocomp.common import logger
from autocomp.search.prob import Prob
from autocomp.agents.llm_agent import LLMAgent
from autocomp.search.code_repo import CodeCandidate

class LLMEnsemble:
    def __init__(self, llms: list[LLMAgent]):
        self.llms = llms

    def __repr__(self):
        return f"LLMEnsemble({self.llms})"
    
    def divide_work(self, num_to_gen: int):
        num_agents = len(self.llms)
        jobs_per_worker = num_to_gen // num_agents
        extra_jobs = num_to_gen % num_agents  # These need to be distributed among the first few workers
        
        job_assignments = []
        for i in range(num_agents):
            num_assigned = jobs_per_worker + (1 if i < extra_jobs else 0)  # Give one extra job to the first 'extra_jobs' workers
            job_assignments.append(num_assigned)

        return job_assignments

    def _run_parallel(self, fn_and_args: list[tuple]) -> list:
        """Run a list of (callable, *args) in parallel threads, returning results in order.
        
        Each element of fn_and_args is a tuple of (fn, arg1, arg2, ...).
        Results are returned in the same order as the input list.
        """
        if not fn_and_args:
            return []
        # Single agent – skip thread overhead
        if len(fn_and_args) == 1:
            fn, *args = fn_and_args[0]
            return [fn(*args)]

        results = [None] * len(fn_and_args)
        with ThreadPoolExecutor(max_workers=len(fn_and_args)) as executor:
            future_to_idx = {}
            for idx, (fn, *args) in enumerate(fn_and_args):
                future = executor.submit(fn, *args)
                future_to_idx[future] = idx
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()  # will re-raise any exception
        return results

    def get_opt_menu_options(self, prob=None):
        return self.llms[0].get_opt_menu_options(prob)

    def analyze_code(self, candidate: CodeCandidate, num_to_gen: int, save_dir: pathlib.Path, save_str: str, prob: Prob = None) -> list[str]:
        num_to_gen_per_agent = self.divide_work(num_to_gen)
        tasks = []
        for i, llm in enumerate(self.llms):
            if num_to_gen_per_agent[i] > 0:
                tasks.append((llm.analyze_code, candidate, num_to_gen_per_agent[i], save_dir, save_str+"_"+self.llms[i].llm_client.model, prob))

        responses = []
        for result in self._run_parallel(tasks):
            responses.extend(result)
        return responses

    def propose_optimizations_parallel(self, candidate_lst: list[CodeCandidate], num_plans: int, save_dir: pathlib.Path, save_strs: list[str], 
                              prob: Prob,
                              force_opt_menu_lst: int = None, 
                              prompt_end: str = "", 
                              analysis_lst: list[str] = None, 
                              shuffle_opts: bool = False, 
                              give_score_feedback: float = 1,
                              give_util_feedback: float = 0,
                              give_hw_feedback: float = 1,
                              include_ancestors: bool = True,
                              plan_icl_examples: bool = False,
                              cur_iter: int = None,
                              num_iters: int = None,
                              dropout_menu_options: float = 1,
                              translate: bool = False,
                             ) -> list[CodeCandidate]:
        num_to_gen_per_agent = self.divide_work(num_plans)
        tasks = []
        for i, llm in enumerate(self.llms):
            if num_to_gen_per_agent[i] > 0:
                this_model_save_strs = [save_str+"_"+self.llms[i].llm_client.model for save_str in save_strs]
                tasks.append((llm.propose_optimizations_parallel, candidate_lst, num_to_gen_per_agent[i], save_dir, this_model_save_strs,
                                    prob,
                                    force_opt_menu_lst, 
                                    prompt_end, 
                                    analysis_lst, 
                                    shuffle_opts, 
                                    give_score_feedback,
                                    give_util_feedback,
                                    give_hw_feedback,
                                    include_ancestors,
                                    plan_icl_examples,
                                    cur_iter,
                                    num_iters,
                                    dropout_menu_options,
                                    translate,
                                    ))

        # Generate optimized menu options per code candidate.
        # Distribute work across models, similarly to how we divide plans, making a total of BEAM_SIZE calls.
        # In each iteration, each candidate is processed by one LLMAgent in the ensemble.
        # Then, the proposed menu options for that candidate are shared with the other LLMAgents in the ensemble.
        if self.llms[0].menu_strategy is not None and not translate:
            num_to_handle_per_agent = self.divide_work(len(candidate_lst))
            tasks_new_menus = []
            curr_candidate_idx = 0
            for i, llm in enumerate(self.llms):
                n = num_to_handle_per_agent[i]
                if n > 0:
                    candidate_subset = candidate_lst[curr_candidate_idx:curr_candidate_idx+n]
                    tasks_new_menus.append((llm.propose_new_menu_parallel, prob, candidate_subset))
                    curr_candidate_idx += n

            new_menu = {}
            for result in self._run_parallel(tasks_new_menus):
                new_menu.update(result)

            for llm in self.llms:
                llm.update_new_menu_cache(new_menu)
            total_new = sum(len(v) for v in new_menu.values())
            logger.info("Dynamically generated %d new menu options across %d candidates.", total_new, len(new_menu))

        cands = []
        for result in self._run_parallel(tasks):
            cands.extend(result)
        return cands

    def implement_code_parallel(self, candidate_lst: list[CodeCandidate], num_samples: int, save_dir: pathlib.Path, save_strs: list[str]=None, code_icl_examples: bool = True, prob: Prob = None) -> list[CodeCandidate]:
        num_to_gen_per_agent = self.divide_work(num_samples)
        tasks = []
        for i, llm in enumerate(self.llms):
            if num_to_gen_per_agent[i] > 0:
                this_model_save_strs = [save_str+"_"+self.llms[i].llm_client.model for save_str in save_strs]
                tasks.append((llm.implement_code_parallel, candidate_lst, num_to_gen_per_agent[i], save_dir, this_model_save_strs, code_icl_examples, prob))

        cands = []
        for result in self._run_parallel(tasks):
            cands.extend(result)
        return cands

    def combine_candidates(self, candidates: list[CodeCandidate], num_samples: int, save_dir: pathlib.Path, save_str: str="", prob: Prob = None) -> list[CodeCandidate]:
        num_to_gen_per_agent = self.divide_work(num_samples)
        tasks = []
        for i, llm in enumerate(self.llms):
            if num_to_gen_per_agent[i] > 0:
                tasks.append((llm.combine_candidates, candidates, num_to_gen_per_agent[i], save_dir, save_str+"_"+self.llms[i].llm_client.model, prob))

        cands = []
        for result in self._run_parallel(tasks):
            cands.extend(result)
        return cands

    def reimplement_failed_code_parallel(self, candidate_lst: list[CodeCandidate], num_samples: int, save_dir: pathlib.Path, save_strs: list[str]=None, prob: Prob = None) -> list[CodeCandidate]:
        """
        Reimplement failed implementations using stdout/stderr from the last attempt.
        """
        num_to_gen_per_agent = self.divide_work(num_samples)
        tasks = []
        for i, llm in enumerate(self.llms):
            if num_to_gen_per_agent[i] > 0:
                this_model_save_strs = [save_str+"_"+self.llms[i].llm_client.model for save_str in save_strs]
                tasks.append((llm.reimplement_failed_code_parallel, candidate_lst, num_to_gen_per_agent[i], save_dir, this_model_save_strs, prob))

        cands = []
        for result in self._run_parallel(tasks):
            cands.extend(result)
        return cands

    def score_translation_completeness(self, original_code: str, candidates: list[CodeCandidate], prob: Prob) -> list[float]:
        """Score translation completeness using the first agent in the ensemble."""
        return self.llms[0].score_translation_completeness(original_code, candidates, prob=prob)

    @property
    def cache_dir(self):
        return self._cache_dir if hasattr(self, "_cache_dir") else None

    @cache_dir.setter
    def cache_dir(self, path):
        self._cache_dir = path
        for agent in self.llms:
            if hasattr(agent, "cache_dir"):
                agent.cache_dir = path

import pathlib

from autocomp.search.prob import Prob
from autocomp.search.llm_agent import LLMAgent
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

    def get_opt_menu_options(self):
        return self.llms[0].get_opt_menu_options()

    def analyze_code(self, candidate: CodeCandidate, num_to_gen: int, save_dir: pathlib.Path, save_str: str) -> list[str]:
        num_to_gen_per_agent = self.divide_work(num_to_gen)
        responses = []
        for i, llm in enumerate(self.llms):
            if num_to_gen_per_agent[i] > 0:
                this_agent_resps = llm.analyze_code(candidate, num_to_gen_per_agent[i], save_dir, save_str+"_"+self.llms[i].llm_client.model)
                responses.extend(this_agent_resps)
        return responses

    def propose_optimizations_parallel(self, candidate_lst: list[CodeCandidate], num_plans: int, save_dir: pathlib.Path, save_strs: list[str], 
                              prob: Prob,
                              force_opt_menu_lst: int = None, 
                              prompt_end: str = "", 
                              analysis_lst: list[str] = None, 
                              shuffle_opts: bool = False, 
                              give_score_feedback: float = 1,
                              give_util_feedback: float = 0,
                              give_spad_acc_feedback: float = 1,
                              include_ancestors: bool = True,
                              plan_icl_examples: bool = False,
                              cur_iter: int = None,
                              num_iters: int = None,
                              dropout_menu_options: float = 1,
                              translate: bool = False,
                             ) -> list[CodeCandidate]:
        num_to_gen_per_agent = self.divide_work(num_plans)
        cands = []
        for i, llm in enumerate(self.llms):
            if num_to_gen_per_agent[i] > 0:
                this_model_save_strs = [save_str+"_"+self.llms[i].llm_client.model for save_str in save_strs]
                this_agent_resps = llm.propose_optimizations_parallel(candidate_lst, num_to_gen_per_agent[i], save_dir, this_model_save_strs, 
                                    prob=prob,
                                    force_opt_menu_lst=force_opt_menu_lst, 
                                    prompt_end=prompt_end, 
                                    analysis_lst=analysis_lst, 
                                    shuffle_opts=shuffle_opts, 
                                    give_score_feedback=give_score_feedback,
                                    give_util_feedback=give_util_feedback,
                                    give_spad_acc_feedback=give_spad_acc_feedback,
                                    include_ancestors=include_ancestors,
                                    plan_icl_examples=plan_icl_examples,
                                    cur_iter=cur_iter,
                                    num_iters=num_iters,
                                    dropout_menu_options=dropout_menu_options,
                                    translate=translate,
                                    )
                cands.extend(this_agent_resps)
        return cands

    def propose_optimizations(self, candidate: CodeCandidate, num_plans: int, save_dir: pathlib.Path, save_str: str, 
                                prob: Prob,
                                force_opt_menu: int = None, 
                                prompt_end: str = "", 
                                analysis: str = "", 
                                shuffle_opts: bool = False, 
                                give_score_feedback: float = 1,
                                give_util_feedback: float = 0,
                                give_spad_acc_feedback: float = 1,
                                include_ancestors: bool = True,
                                plan_icl_examples: bool = False,
                                cur_iter: int = None,
                                num_iters: int = None,
                                dropout_menu_options: float = 1,
                                translate: bool = False,
                                ) -> list[CodeCandidate]:
        num_to_gen_per_agent = self.divide_work(num_plans)
        cands = []
        for i, llm in enumerate(self.llms):
            if num_to_gen_per_agent[i] > 0:
                this_agent_resps = llm.propose_optimizations(candidate, num_to_gen_per_agent[i], save_dir, save_str+"_"+self.llms[i].llm_client.model, 
                                    prob=prob,
                                    force_opt_menu=force_opt_menu, 
                                    prompt_end=prompt_end, 
                                    analysis=analysis, 
                                    shuffle_opts=shuffle_opts, 
                                    give_score_feedback=give_score_feedback,
                                    give_util_feedback=give_util_feedback,
                                    give_spad_acc_feedback=give_spad_acc_feedback,
                                    include_ancestors=include_ancestors,
                                    plan_icl_examples=plan_icl_examples,
                                    cur_iter=cur_iter,
                                    num_iters=num_iters,
                                    dropout_menu_options=dropout_menu_options,
                                    translate=translate,
                                    )
                cands.extend(this_agent_resps)
        return cands

    def implement_code_parallel(self, candidate_lst: list[CodeCandidate], num_samples: int, save_dir: pathlib.Path, save_strs: list[str]=None, code_icl_examples: bool = True, prob: Prob = None) -> list[CodeCandidate]:
        num_to_gen_per_agent = self.divide_work(num_samples)
        cands = []
        for i, llm in enumerate(self.llms):
            if num_to_gen_per_agent[i] > 0:
                this_model_save_strs = [save_str+"_"+self.llms[i].llm_client.model for save_str in save_strs]
                this_agent_resps = llm.implement_code_parallel(candidate_lst, num_to_gen_per_agent[i], save_dir, save_strs=this_model_save_strs, code_icl_examples=code_icl_examples, prob=prob)
                cands.extend(this_agent_resps)
        return cands

    def implement_code(self, candidate: CodeCandidate, num_samples: int, save_dir: pathlib.Path, save_str: str="", code_icl_examples: bool = True, prob: Prob = None) -> list[CodeCandidate]:
        num_to_gen_per_agent = self.divide_work(num_samples)
        cands = []
        for i, llm in enumerate(self.llms):
            if num_to_gen_per_agent[i] > 0:
                this_agent_resps = llm.implement_code(candidate, num_to_gen_per_agent[i], save_dir, save_str+"_"+self.llms[i].llm_client.model, code_icl_examples=code_icl_examples, prob=prob)
                cands.extend(this_agent_resps)
        return cands

    def combine_candidates(self, candidates: list[CodeCandidate], num_samples: int, save_dir: pathlib.Path, save_str: str="") -> list[CodeCandidate]:
        num_to_gen_per_agent = self.divide_work(num_samples)
        cands = []
        for i, llm in enumerate(self.llms):
            if num_to_gen_per_agent[i] > 0:
                this_agent_resps = llm.combine_candidates(candidates, num_to_gen_per_agent[i], save_dir, save_str+"_"+self.llms[i].llm_client.model)
                cands.extend(this_agent_resps)
        return cands

    def reimplement_failed_code_parallel(self, candidate_lst: list[CodeCandidate], num_samples: int, save_dir: pathlib.Path, save_strs: list[str]=None, prob: Prob = None) -> list[CodeCandidate]:
        """
        Reimplement failed code candidates using stdout/stderr from the last attempt.
        This method is parallelized across multiple LLM agents.
        """
        num_to_gen_per_agent = self.divide_work(num_samples)
        cands = []
        for i, llm in enumerate(self.llms):
            if num_to_gen_per_agent[i] > 0:
                this_model_save_strs = [save_str+"_"+self.llms[i].llm_client.model for save_str in save_strs]
                this_agent_resps = llm.reimplement_failed_code_parallel(candidate_lst, num_to_gen_per_agent[i], save_dir, save_strs=this_model_save_strs, prob=prob)
                cands.extend(this_agent_resps)
        return cands

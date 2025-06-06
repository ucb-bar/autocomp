import pathlib
import random

from autocomp.common import logger, LLMClient, llm_utils
from autocomp.search.prob import Prob
from autocomp.search.code_repo import CodeCandidate, copy_candidate
from autocomp.search.annotate_perf import GemminiCode
from prompts import isa_prompt_conv, isa_prompt_admm
from prompts.opt_system import plan_prompt, gemmini_rules, tiling_example, if_example, if_example_matmul

prob_macs_map = {
"exo": [
    512 * 512 * 512,
    12544 * 64 * 256,
    12544 * 256 * 64,
    3136 * 128 * 512,
    3136 * 512 * 128,
    784 * 256 * 1024,
],
"exo-conv": [
    462422016,
    462422016,
    462422016,
],
"admm-multifunction": [
    (12*4*1+12*4*1+12*12*1)*5,
    (4*12*1+4*4*1+4*12*1+12*12*1)*10,
]
}

def extract(code_str: str) -> str:
    """
    Takes LLM-generated code and extracts the "test" wrapper function

    for example:
    '''
    void test(Kinf, r, K_r) {
        config_ex(WEIGHT_STATIONARY,  NO_ACTIVATION, true, false);
        config_st(1 * sizeof(float)); // output K_r has 1 column in DRAM
        ...
        mvout(K_r + 0x8, K_r_acc_addr + 8, 1, 4); // mvout the third 4x1 block of K_r
        fence();
    }
    '''
    """
    # Remove the function wrapper and return only the body of the function
    if not code_str:
        return code_str
    from_void_test_str = code_str[code_str.find("void test"):]
    # end = from_void_test_str.rfind('}\n')
    # Iterate through all characters until the second time matching curly braces have been found
    open_braces = 0
    in_comment = False
    in_single_line_comment = False
    for i, char in enumerate(from_void_test_str):
        if from_void_test_str[i:i+2] == '/*':
            in_comment = True
        elif from_void_test_str[i:i+2] == '*/':
            in_comment = False
        if from_void_test_str[i:i+2] == '//':
            in_single_line_comment = True
        elif char == '\n':
            in_single_line_comment = False
        if in_comment or in_single_line_comment:
            continue
        if char == '{':
            open_braces += 1
        elif char == '}':
            open_braces -= 1
            if open_braces == 0:
                end = i
                break
    try:
        body = from_void_test_str[:end+1]
    except:
        return from_void_test_str
    return body
    # end = from_void_test_str.find('fence();')
    # find_last_curly = 0
    # last_idx = len(from_void_test_str)
    # while end + find_last_curly < len(from_void_test_str):
    #     if from_void_test_str[end + find_last_curly] == '}':
    #         last_idx = end + find_last_curly
    #     find_last_curly += 1
    # body = from_void_test_str[:last_idx+1]
    # return body
    # body = code_str[start:end].split("\n")
    # new_body = []
    # for line in body:
    #     new_body.append(line.strip())
    # return "\n".join(new_body)

class LLMAgent:
    """
    A mock-up of the LLM used to propose, evaluate, and implement optimizations.
    """
    def __init__(self, model):
        self.llm_client = LLMClient(model)

    def get_opt_menu_options(self):
        # Mock-up method to get the optimization menu options
        return ["Opt 1", "Opt 2", "Opt 3"]

    def analyze_code(self, candidate: CodeCandidate, num_to_gen: int, save_dir: pathlib.Path, save_str: str) -> list[str]:
        return ["Analysis 1", "Analysis 2", "Analysis 3"]

    def propose_optimizations_parallel(self, candidate_lst: list[CodeCandidate], num_plans: int, save_dir: pathlib.Path, save_strs: list[str], 
                                force_opt_menu_lst: int = None, 
                                prompt_end: str = "", 
                                analysis_lst: list[str] = None, 
                                shuffle_opts: bool = False, 
                                give_score_feedback: bool = True,
                                include_ancestors: bool = True,
                                plan_icl_examples: bool = False,
                                cur_iter: int = None,
                                num_iters: int = None,
                                dropout_menu_options: float = 1,
                                ) -> list[CodeCandidate]:
        raise NotImplementedError

    def propose_optimizations(self, candidate: CodeCandidate, num_plans: int, save_dir: pathlib.Path, save_str: str, 
                              force_opt_menu: int = None, 
                              prompt_end: str = "", 
                              analysis: str = "", 
                              shuffle_opts: bool = False, 
                              give_score_feedback: bool = True,
                              include_ancestors: bool = True,
                              plan_icl_examples: bool = False,
                              cur_iter: int = None,
                              num_iters: int = None,
                             ) -> list[CodeCandidate]:
        # Mock-up method: return some optimization plans
        return [CodeCandidate(candidate, "Plan A", None), CodeCandidate(candidate, "Plan B", None), CodeCandidate(candidate, "Plan C", None)]

    def implement_code_parallel(self, candidate_lst: list[CodeCandidate], num_samples: int, save_dir: pathlib.Path, save_strs: list[str] = None, code_icl_examples: bool = True) -> list[CodeCandidate]:
        raise NotImplementedError

    def implement_code(self, candidate: CodeCandidate, num_samples: int, save_dir: pathlib.Path) -> str:
        # Mock-up method to implement the code
        assert num_samples > 0
        candidates = [candidate]
        logger.debug(f"Implementing plan:\n{candidate.plan}\nOriginal code:\n{candidate.parent.code}")
        for i in range(num_samples):
            if i == 0:
                candidates[i].code = "Optimized code 0"
            else:
                candidates.append(CodeCandidate(candidate.parent, candidate.plan, f"Optimized code {i}"))
            logger.debug(f"Sample {i+1}: {candidate.code}")
        return candidates

    def combine_candidates(self, candidates: list[CodeCandidate], num_samples: int, save_dir: pathlib.Path, save_str: str="") -> list[CodeCandidate]:
        # Mock-up method to combine the candidates
        combined_candidates = []
        for i in range(num_samples):
            combined_candidates.append(CodeCandidate(candidates, f"Combined parents", f"Combined code {i}"))
        return combined_candidates

class GemminiLLMAgent(LLMAgent):
    def __init__(self, model, pe_dim):
        super().__init__(model)
        self.pe_dim = pe_dim

    def __repr__(self):
        return f"GemminiLLMAgent({self.llm_client.model}, {self.pe_dim})"

    def get_opt_menu_options(self, prob: Prob):
        if "admm" in prob.prob_type:
            return [
                "remove unnecessary code",
                "simplify arithmetic and propagate constants to simplify expressions",
                "merge instructions",
                "merge high-level operations",
                "reorder operations or blocks of operations",
                "move CPU-based computation to the accelerator",
                "add or subtract a matrix using the bias",
                "hoist redundant operations out of loops",
                "substitute operations with equivalent operations that are faster",
                "pipeline operations to better overlap computation and data movement",
                "eliminate data dependencies and fence operations",
                "minimize data movement",
                "minimize loop overhead",
                "other methods not listed here.",
            ]
        else: # gemm and conv
            return [
                "modify loop tiling",
                "loop reordering",
                "split loops",
                "fuse loops",
                "simplify arithmetic and propagate constants to simplify expressions",
                "reorder computations or blocks of computations",
                "loop unrolling",
                "double buffering",
                "move more data to the scratchpad in a more outer loop to increase data reuse",
                "spread data throughout the scratchpad rather than loading to the same location repeatedly",
                "load data to the scratchpad across outer loop iterations and use if statements to prevent redundant loads on loops inner to those",
                "hoist redundant operations out of loops",
                "substitute operations with equivalent operations that are faster",
                "pipeline operations to better overlap computation and data movement",
                "minimize data movement",
                "minimize loop overhead",
                "other methods not listed here.",
            ]
        # "remove redundant code",
        # "compute common sub-expressions ahead of time",
        # "merge instructions",
        # "merge high-level operations",
        # "move CPU-based computation to the accelerator",

    def analyze_code(self, candidate: CodeCandidate, num_to_gen: int, save_dir: pathlib.Path, save_str: str) -> list[str]:
        """
        Generate an analysis of current code based on performance feedback
        """
        if self.pe_dim == 4:
            prompt_text = "The Gemmini accelerator's ISA is as follows:" + isa_prompt_admm.PROMPT(self.pe_dim) + "\n"
        elif self.pe_dim == 16:
            prompt_text = "The Gemmini accelerator's ISA is as follows:" + isa_prompt_conv.PROMPT(self.pe_dim) + "\n"
        prompt_text += "Elements in the scratchpad are 1 byte, and elements in the accumulator are 4 bytes.\n"
        prompt_text += "The scratchpad size is 256KB and the accumulator size is 64KB. The systolic array is 16 by 16.\n"
        prompt_text += "The original code is as follows:\n" + candidate.code + "\n"
        prompt_text += "You are an optimizing compiler that produces high-performance Gemmini code. Based on this information, analyze the code and identify the single most impactful bottleneck increasing cycle count."
        # Save prompt
        prompt_path = f"prompt{'' if not save_str else '_' + save_str}"
        prompt_path += ".txt"
        prompt_path = save_dir / prompt_path
        with open(prompt_path, "w") as f:
            f.write(prompt_text)
        # Call LLM to generate responses
        messages = [{"role": "user", "content": prompt_text}]
        temperature = 1
        responses = self.llm_client.chat(
            messages=messages,
            num_candidates=num_to_gen,  # number of candidates,
            temperature=temperature
        )

        # Save responses
        for c_i, c in enumerate(responses):
            path = f"analyze{'' if not save_str else '_' + save_str}"
            path += "_" + str(c_i) + ".txt"
            path = save_dir / path
            with open(path, "w") as f:
                f.write(c)
            logger.debug("Saved select_optimization response to %s", path)

        return responses

    # def select_optimization(self, candidate: CodeCandidate, num_to_gen: int, save_dir: pathlib.Path, save_str: str, iteration: int, num_interations: int, analysis_str: str = "") -> str:
    #     """
    #     Select an optimization to apply from the menu
    #     """
    #     # Construct the prompt
    #     prompt_text: str = "The Gemmini accelerator's ISA is as follows:" + isa_prompt_conv.TEMPLATE_TEXT + "\n"
    #     # Add the previously iterated plans and code to the prompt
    #     parents_prompt = ""
    #     cur_cand: CodeCandidate = candidate
    #     while cur_cand is not None:
    #         # Go up to each parent and append to front of prompt
    #         # annotated_code = GemminiCode(cur_cand.code, self.pe_dim).annotate_perf()
    #         if len(cur_cand.feedback) > 0:
    #             parents_prompt = "\n".join(cur_cand.feedback) + "\n" + parents_prompt
    #         if cur_cand.score is not None:
    #             parents_prompt = f"The latency of this code was: {cur_cand.score} cycles.\n" + parents_prompt
    #         if cur_cand.plan is not None:
    #             parents_prompt = "\nNext, we applied this <optimization> to the code:\n" + cur_cand.opt_selection + "\nThe generated code was:\n" + cur_cand.code + "\n" + parents_prompt
    #         else:
    #             parents_prompt = "The original unoptimized code was:\n" + cur_cand.code + "\n" + parents_prompt
    #         cur_cand = cur_cand.parent
    #     prompt_text += parents_prompt
    #     if analysis_str:
    #         prompt_text += analysis_str + "\n"
    #     prompt_text += "It is currently phase " + str(iteration) + " out of " + str(num_interations) + " optimization phases.\n"
    #     prompt_text += "You are an optimizing compiler that produces high-performance Gemmini code. Based on this information, select an <optimization> to apply.\n"
    #     menu_options_text = "<optimizations>\n"
    #     for i, opt in enumerate(self.get_opt_menu_options()):
    #         menu_options_text += f"{i+1}. {opt}\n"
    #     prompt_text += menu_options_text

    #     # Save prompt
    #     prompt_path = f"prompt{'' if not save_str else '_' + save_str}"
    #     prompt_path += ".txt"
    #     prompt_path = save_dir / prompt_path
    #     with open(prompt_path, "w") as f:
    #         f.write(prompt_text)

    #     # Call LLM to generate responses
    #     messages = {"role": "user", "content": prompt_text}
    #     temperature = 1
    #     responses = self.llm_client.chat(
    #         messages=messages,
    #         num_candidates=num_to_gen,  # number of candidates,
    #         temperature=temperature
    #     )

    #     # Save responses
    #     for c_i, c in enumerate(responses):
    #         path = f"select{'' if not save_str else '_' + save_str}"
    #         path += "_" + str(c_i) + ".txt"
    #         path = save_dir / path
    #         with open(path, "w") as f:
    #             f.write(c)
    #         logger.debug("Saved select_optimization response to %s", path)

    #     return responses

    # def gen_plan_from_opt(self, candidate: CodeCandidate, num_to_gen: int, opt_selection: str, save_dir: pathlib.Path, save_str: str) -> str:
    #     # Construct the prompt
    #     prompt_text = "The Gemmini accelerator's ISA is as follows:" + isa_prompt_conv.TEMPLATE_TEXT + "\n"
    #     prompt_text += "The original code was:" + candidate.code + "\n"
    #     prompt_text += "You have selected the following <optimization> to apply:\n" + opt_selection + "\n"
    #     prompt_text += "You are an optimizing compiler that produces high-performance Gemmini code. Based on this information, come up with a plan of how you would implement the <optimization> described.\n"

    #     # Save prompt
    #     prompt_path = f"prompt{'' if not save_str else '_' + save_str}"
    #     prompt_path += ".txt"
    #     prompt_path = save_dir / prompt_path
    #     with open(prompt_path, "w") as f:
    #         f.write(prompt_text)

    #     # Call LLM to generate responses
    #     messages = {"role": "user", "content": prompt_text}
    #     temperature = 1
    #     responses = self.llm_client.chat(
    #         messages=messages,
    #         num_candidates=num_to_gen,  # number of candidates,
    #         temperature=temperature
    #     )

    #     # Save responses
    #     for c_i, c in enumerate(responses):
    #         path = f"plan{'' if not save_str else '_' + save_str}"
    #         path += "_" + str(c_i) + ".txt"
    #         path = save_dir / path
    #         with open(path, "w") as f:
    #             f.write(c)
    #         logger.debug("Saved optimization plan to %s", path)

    #     return responses

    def _get_propose_optimizations_prompt(self, candidate: CodeCandidate,
                                          prob: Prob,
                                          force_opt_menu: int, 
                                          prompt_end: str, 
                                          analysis: str, 
                                          shuffle_opts: bool, 
                                          give_score_feedback: float,
                                          give_util_feedback: float,
                                          give_spad_acc_feedback: float,
                                          include_ancestors: bool,
                                          plan_icl_examples: bool,
                                          cur_iter: int,
                                          num_iters: int,
                                          dropout_menu_options: float,
                                         ) -> list[str]:
        # Select which menu options will appear
        plan_prompt_texts = plan_prompt.PROMPT(self.pe_dim)
        menu_options_text = ""
        opt_lst = self.get_opt_menu_options(prob)
        if dropout_menu_options < 1 and not force_opt_menu:
            opt_lst = [opt for opt in opt_lst if random.random() < dropout_menu_options]
        if shuffle_opts:
            random.shuffle(opt_lst)
        include_util_feedback = random.random() < give_util_feedback
        include_score_feedback = random.random() < give_score_feedback
        include_spad_acc_feedback = random.random() < give_spad_acc_feedback

        parents_prompt = ""
        cur_cand = candidate
        while cur_cand is not None:
            # Go up to each parent and append to front of prompt
            # annotated_code = GemminiCode(cur_cand.code, self.pe_dim).annotate_perf()
            if include_spad_acc_feedback:
                parents_prompt = "\n".join(cur_cand.spad_acc_stats) + "\n" + parents_prompt
            if include_util_feedback and (cur_cand.score is not None):
                macs = prob_macs_map[prob.prob_type][prob.prob_id]
                theoretical_min_cycles = macs / (self.pe_dim ** 2)
                util = theoretical_min_cycles / cur_cand.score * 100
                parents_prompt = f"The utilization of this code was {round(util)}%.\n" + parents_prompt
            if include_score_feedback and (cur_cand.score is not None):
                parents_prompt = f"The latency of this code was {cur_cand.score} cycles.\n" + parents_prompt
            if not include_ancestors:
                parents_prompt = "\nThe original unoptimized code was:\n" + cur_cand.code + "\n" + parents_prompt
                break # No need to go up past the immediate parent
            elif cur_cand.plan is not None:
                parents_prompt = "\nNext, we applied this plan to the code:\n" + cur_cand.plan + "\nThe generated code was:\n" + cur_cand.code + "\n" + parents_prompt
            else:
                parents_prompt = "The original unoptimized code was:\n" + cur_cand.code + "\n" + parents_prompt
            cur_cand = cur_cand.parent

        if analysis:
            parents_prompt += "\n" + analysis

        # Initialize the prompt with the parents info
        # prompt_text = ""
        if self.pe_dim == 4:
            prompt_text = "\nThe Gemmini accelerator's ISA is as follows:" + isa_prompt_admm.PROMPT(self.pe_dim)
        elif self.pe_dim == 16:
            prompt_text = "\nThe Gemmini accelerator's ISA is as follows:" + isa_prompt_conv.PROMPT(self.pe_dim)
        if plan_icl_examples:
            if "modify loop tiling" in opt_lst:
                prompt_text += tiling_example.PROMPT()
            if "move more data to the scratchpad in a more outer loop to increase data reuse" in opt_lst:
                prompt_text += if_example.PROMPT()
            elif "spread data throughout the scratchpad rather than loading to the same location repeatedly" in opt_lst:
                prompt_text += if_example.PROMPT()
            elif "load data to the scratchpad across outer loop iterations and use if statements to prevent redundant loads on loops inner to those" in opt_lst:
                prompt_text += if_example.PROMPT()
        prompt_text += parents_prompt

        # Now add the actual planning prompt
        for i, opt in enumerate(opt_lst):
            menu_options_text += f"{i+1}. {opt}\n"
        prompt_text += plan_prompt_texts["PRE_OPT_TEXT"] + "\n" + menu_options_text + "\n" + plan_prompt_texts["POST_OPT_TEXT"]
        if force_opt_menu:
            prompt_text += "Explain how to apply <optimization> " + str(force_opt_menu) + ": '" + opt_lst[force_opt_menu-1] + "' to the above code to reduce cycle count, and explain how it will improve performance."
        else:
            prompt_text += plan_prompt_texts["FINAL_TEXT"]

        if prompt_end:
            logger.debug("Appended the following as prompt_end: '%s'", prompt_end)
            prompt_text += "\n" + prompt_end
        return prompt_text

    def propose_optimizations_parallel(self, candidate_lst: list[CodeCandidate], num_plans: int, save_dir: pathlib.Path, save_strs: list[str], 
                              prob: Prob,
                              force_opt_menu_lst: int = None, 
                              prompt_end: str = "", 
                              analysis_lst: list[str] = None, 
                              shuffle_opts: bool = False, 
                              give_score_feedback: float = 1.0,
                              give_util_feedback: float = 0.0,
                              give_spad_acc_feedback: float = 1.0,
                              include_ancestors: bool = True,
                              plan_icl_examples: bool = False,
                              cur_iter: int = None,
                              num_iters: int = None,
                              dropout_menu_options: float = 1,
                             ) -> list[CodeCandidate]:
        """
        dropout_menu_options: probability of keeping each menu option
        """
        loaded_plans = []
        not_found = False
        for c_i in range(len(candidate_lst)):
            this_cand_loaded_plans = []
            save_str = save_strs[c_i]
            for s_i in range(num_plans):
                path = f"plan{'' if not save_str else '_' + save_str}"
                if force_opt_menu_lst is not None:
                    path += "_" + str(force_opt_menu_lst[c_i])
                path += "_" + str(s_i) + ".txt"
                path = save_dir / path
                if path.exists():
                    with open(path, "r") as f:
                        plan = f.read()
                        logger.debug("Loaded optimization plan from %s", path)
                        this_cand_loaded_plans.append(plan)
                else:
                    not_found = True
                    break
            if not_found:
                break
            loaded_plans.append(this_cand_loaded_plans)
        else:
            loaded_cands = []
            for c_i, this_cand_loaded_plans in enumerate(loaded_plans):
                for plan in this_cand_loaded_plans:
                    loaded_cands.append(CodeCandidate(candidate_lst[c_i], plan, None, plan_gen_model=self.llm_client.model))
            logger.info("Loaded %d optimization plans rather than generating new ones", len(loaded_cands))
            return loaded_cands

        if dropout_menu_options < 1 or (0 < give_score_feedback < 1) or (0 < give_util_feedback < 1) or (0 < give_spad_acc_feedback < 1):
            num_unique_prompts_per_cand = num_plans
        else:
            num_unique_prompts_per_cand = 1
        msgs_lst = []
        for c_i, candidate in enumerate(candidate_lst):
            save_str = save_strs[c_i]
            for p in range(num_unique_prompts_per_cand):
                # Add the previously iterated plans and code to the prompt
                analysis = "" if analysis_lst is None else analysis_lst[c_i]
                force_opt_menu = None if force_opt_menu_lst is None else force_opt_menu_lst[c_i]
                prompt_text = self._get_propose_optimizations_prompt(candidate, prob, force_opt_menu, prompt_end, analysis, shuffle_opts,
                                                                    give_score_feedback, give_util_feedback, give_spad_acc_feedback, include_ancestors, plan_icl_examples, cur_iter, num_iters,
                                                                    dropout_menu_options)

                # Save full prompt
                prompt_path = f"prompt{'' if not save_str else '_' + save_str}"
                if force_opt_menu_lst is not None:
                    prompt_path += "_" + str(force_opt_menu_lst[c_i])
                else:
                    prompt_path += "_" + str(p)
                prompt_path += ".txt"
                prompt_path = save_dir / prompt_path

                with open(prompt_path, "w") as f:
                    f.write(prompt_text)

                messages = [
                    # {"role": "system", "content": TEMPLATE_SYS},
                    {"role": "user", "content": prompt_text},
                ]
                msgs_lst.append(messages)

        temperature = 1
        candidates_to_gen = num_plans // num_unique_prompts_per_cand

        extended_responses = self.llm_client.chat_async(
            msgs_lst=msgs_lst,
            num_candidates=candidates_to_gen,  # number of candidates,
            temperature=temperature
        )
        # Need to sort the responses back into a flattened list for each parent candidate
        responses = [[] for _ in range(len(candidate_lst))]
        for r_i, per_prompt_responses in enumerate(extended_responses):
            c_i = r_i // num_unique_prompts_per_cand
            responses[c_i].extend(per_prompt_responses)

        for c_i in range(len(responses)):
            save_str = save_strs[c_i]
            for s_i in range(num_plans):
                path = f"plan{'' if not save_str else '_' + save_str}"
                if force_opt_menu_lst is not None:
                    path += "_" + str(force_opt_menu_lst[c_i])
                path += "_" + str(s_i) + ".txt"
                path = save_dir / path
                with open(path, "w") as f:
                    f.write(responses[c_i][s_i])
                logger.debug("Saved optimization plan to %s", path)

        new_cands = []
        for c_i, cand_resps in enumerate(responses):
            for plan in cand_resps:
                new_cands.append(CodeCandidate(candidate_lst[c_i], plan, None, plan_gen_model=self.llm_client.model))
        return new_cands

    def propose_optimizations(self, candidate: CodeCandidate, num_plans: int, save_dir: pathlib.Path, save_str: str, 
                              prob: Prob,
                              force_opt_menu: int = None, 
                              prompt_end: str = "", 
                              analysis: str = "", 
                              shuffle_opts: bool = False, 
                              give_score_feedback: bool = True,
                              give_util_feedback: bool = False,
                              include_ancestors: bool = True,
                              plan_icl_examples: bool = False,
                              cur_iter: int = None,
                              num_iters: int = None,
                              dropout_menu_options: float = 1,
                             ) -> list[CodeCandidate]:
        """
        dropout_menu_options: probability of keeping each menu option
        """
        loaded_plans = []
        for c_i in range(num_plans):
            path = f"plan{'' if not save_str else '_' + save_str}"
            if force_opt_menu is not None:
                path += "_" + str(force_opt_menu)
            path += "_" + str(c_i) + ".txt"
            path = save_dir / path
            if path.exists():
                with open(path, "r") as f:
                    plan = f.read()
                    logger.debug("Loaded optimization plan from %s", path)
                    loaded_plans.append(plan)
            else:
                break
        else:
            loaded_cands = [CodeCandidate(candidate, plan, None, plan_gen_model=self.llm_client.model) for plan in loaded_plans]
            logger.info("Loaded %d optimization plans rather than generating new ones", num_plans)
            return loaded_cands

        responses = []
        if dropout_menu_options < 1:
            num_unique_prompts_per_cand = num_plans
        else:
            num_unique_prompts_per_cand = 1
        for p in range(num_unique_prompts_per_cand):
            # Add the previously iterated plans and code to the prompt
            prompt_text = self._get_propose_optimizations_prompt(candidate, prob, force_opt_menu, prompt_end, analysis, shuffle_opts,
                                                                 give_score_feedback, give_util_feedback, include_ancestors, plan_icl_examples, cur_iter, num_iters,
                                                                 dropout_menu_options)

            # Save full prompt
            prompt_path = f"prompt{'' if not save_str else '_' + save_str}"
            if force_opt_menu is not None:
                prompt_path += "_" + str(force_opt_menu)
            else:
                prompt_path += "_" + str(p)
            prompt_path += ".txt"
            prompt_path = save_dir / prompt_path

            with open(prompt_path, "w") as f:
                f.write(prompt_text)

            messages = [
                # {"role": "system", "content": TEMPLATE_SYS},
                {"role": "user", "content": prompt_text},
            ]
            temperature = 1
            candidates_to_gen = num_plans // num_unique_prompts_per_cand

            resp = self.llm_client.chat_async(
                msgs_lst=[messages],
                num_candidates=candidates_to_gen,  # number of candidates,
                temperature=temperature
            )[0]
            # resp = self.llm_client.chat(
            #     messages=messages,
            #     num_candidates=candidates_to_gen,  # number of candidates,
            #     temperature=temperature
            # )
            responses.extend(resp)

        for c_i, c in enumerate(responses):
            path = f"plan{'' if not save_str else '_' + save_str}"
            if force_opt_menu is not None:
                path += "_" + str(force_opt_menu)
            path += "_" + str(c_i) + ".txt"
            path = save_dir / path
            with open(path, "w") as f:
                f.write(c)
            logger.debug("Saved optimization plan to %s", path)

        new_cands = [CodeCandidate(candidate, plan, None, plan_gen_model=self.llm_client.model) for plan in responses]
        return new_cands

    def _get_implement_code_prompt(self, candidate: CodeCandidate, code_icl_examples: bool = True) -> list[CodeCandidate]:
        if self.pe_dim == 4:
            prompt_text = "The Gemmini accelerator's ISA is as follows:" + isa_prompt_admm.PROMPT(self.pe_dim)
        elif self.pe_dim == 16:
            prompt_text = "The Gemmini accelerator's ISA is as follows:" + isa_prompt_conv.PROMPT(self.pe_dim)

        prompt_text += "\nThe original code is as follows:\n"
        prompt_text += candidate.parent.code
        prompt_text += "\nYou are an optimizing compiler generating high-performance Gemmini code. Let's optimize the original code based on the following plan:\n"
        prompt_text += candidate.plan

        # # TODO: For certain optimizations, add more context helping it implement prompt correctly.
        # # e.g. for tiling, add examples of how to tile the code.
        if code_icl_examples:
            if "tiling" in candidate.plan:
                prompt_text += "\n" + tiling_example.PROMPT()
            if " gate" in candidate.plan or "Gate" in candidate.plan:
                # prompt_text += "\n" + if_example_matmul.PROMPT()
                prompt_text += "\n" + if_example.PROMPT()

        prompt_text += "\nMake sure to follow these rules:"
        prompt_text += gemmini_rules.PROMPT()

        # prompt_text += "\nRespond with only the optimized code:"
        prompt_text += "Optimized code:"

        return prompt_text

    def implement_code_parallel(self, candidate_lst: list[CodeCandidate], num_samples: int, save_dir: pathlib.Path, save_strs: list[str] = None, code_icl_examples: bool = True) -> list[CodeCandidate]:
        if save_strs is not None:
            assert len(candidate_lst) == len(save_strs)
        loaded_code = []
        code_not_found = False
        for c_i in range(len(candidate_lst)):
            this_cand_loaded_code = []
            save_str = save_strs[c_i]
            for s_i in range(num_samples):
                path = save_dir / f"impl{'' if not save_str else '_' + save_str}_{s_i}_full.txt"
                if path.exists():
                    with open(path, "r") as f:
                        code = extracted_code = extract(f.read())
                        logger.debug("Loaded optimization plan from %s", path)
                        this_cand_loaded_code.append(code)
                else:
                    code_not_found = True
                    break
            if code_not_found:
                break
            loaded_code.append(this_cand_loaded_code)
        else:
            loaded_candidates = []
            for c_i in range(len(candidate_lst)):
                cand = candidate_lst[c_i]
                for s_i in range(num_samples):
                    new_cand = copy_candidate(cand)
                    new_cand.code = loaded_code[c_i][s_i]
                    new_cand.code_gen_model = self.llm_client.model
                    loaded_candidates.append(new_cand)
            logger.info("Loaded %d code implementations rather than generating new ones", len(loaded_candidates))
            return loaded_candidates

        msgs_lst = []
        for c_i in range(len(candidate_lst)):
            prompt_text = self._get_implement_code_prompt(candidate_lst[c_i], code_icl_examples)
            # Save full prompt
            prompt_path = save_dir / f"prompt{'' if not save_strs[c_i] else '_' + save_strs[c_i]}.txt"
            with open(prompt_path, "w") as f:
                f.write(prompt_text)
            messages = [
                # {"role": "system", "content": TEMPLATE_SYS},
                {"role": "user", "content": prompt_text},
            ]
            msgs_lst.append(messages)

        temperature = 1
        responses = self.llm_client.chat_async(
            msgs_lst,
            num_candidates=num_samples,  # number of candidates,
            temperature=temperature
        )

        candidates: list[CodeCandidate] = [] # Flat list of new implemented candidates
        for c_i, cand_responses in enumerate(responses):
            this_plan_cands = []
            for s_i, sample_response in enumerate(cand_responses):
                # Save full response
                full_path = save_dir / f"impl{'' if not save_strs[c_i] else '_' + save_strs[c_i]}_{s_i}_full.txt"
                with open(full_path, "w") as f:
                    f.write(sample_response)
                # Extract just the code
                extracted_code = extract(sample_response)
                if not extracted_code:
                    logger.warning("Failed to extract code from plan %d response %d, full response was %s", 
                                   c_i, s_i, sample_response)
                path = save_dir / f"impl{'' if not save_strs[c_i] else '_' + save_strs[c_i]}_{s_i}.txt"
                with open(path, "w") as f:
                    f.writelines(extracted_code)
                logger.debug("Saved plan %d code impl %d to %s", c_i, s_i, path)
                # Make a copy of the candidate with the new code impl
                new_cand = copy_candidate(candidate_lst[c_i])
                new_cand.code = extracted_code
                new_cand.code_gen_model = self.llm_client.model
                this_plan_cands.append(new_cand)
            candidates.extend(this_plan_cands)
        return candidates

    def implement_code(self, candidate: CodeCandidate, num_samples: int, save_dir: pathlib.Path, save_str: str="") -> list[CodeCandidate]:
        assert num_samples > 0, "Number of samples must be greater than 0"

        loaded_code = []
        for c_i in range(num_samples):
            path = save_dir / f"impl{'' if not save_str else '_' + save_str}_{c_i}.txt"
            if path.exists():
                with open(path, "r") as f:
                    code = f.read()
                    logger.debug("Loaded optimization plan from %s", path)
                    loaded_code.append(code)
            else:
                break
        else:
            logger.info("Loaded %d code implementations rather than generating new ones", num_samples)
            candidate.code = loaded_code[0]
            loaded_candidates = [candidate]
            for c_i in range(1, num_samples):
                new_cand = copy_candidate(candidate)
                new_cand.code = loaded_code[c_i]
                new_cand.code_gen_model = self.llm_client.model
                loaded_candidates.append(new_cand)
            return loaded_candidates

        prompt_text = self._get_implement_code_prompt(candidate)
        # Save full prompt
        prompt_path = save_dir / f"prompt{'' if not save_str else '_' + save_str}.txt"
        with open(prompt_path, "w") as f:
            f.write(prompt_text)

        messages = [
            # {"role": "system", "content": TEMPLATE_SYS},
            {"role": "user", "content": prompt_text},
        ]
        temperature = 1
        responses = self.llm_client.chat(
            messages=messages,
            num_candidates=num_samples,  # number of candidates,
            temperature=temperature
        )

        candidates = []
        for c_i, c in enumerate(responses):
            # Save full response
            full_path = save_dir / f"impl{'' if not save_str else '_' + save_str}_{c_i}_full.txt"
            with open(full_path, "w") as f:
                f.write(c)
            # Extract just the code
            extracted_code = extract(c)
            if not extracted_code:
                logger.warning("Failed to extract code from response %d, full response was %s", c_i, c)
            path = save_dir / f"impl{'' if not save_str else '_' + save_str}_{c_i}.txt"
            with open(path, "w") as f:
                f.writelines(extracted_code)
            logger.debug("Saved code impl %d to %s", c_i, path)

            # Make a copy of the candidate with the new code impl
            new_cand = copy_candidate(candidate)
            new_cand.code = extracted_code
            new_cand.code_gen_model = self.llm_client.model
            candidates.append(new_cand)
        return candidates

    def combine_candidates(self, candidates: list[CodeCandidate], num_samples: int, save_dir: pathlib.Path, save_str: str="") -> list[CodeCandidate]:
        loaded_code = []
        for c_i in range(num_samples):
            path = save_dir / f"combined{'' if not save_str else '_' + save_str}_{c_i}.txt"
            if path.exists():
                with open(path, "r") as f:
                    code = f.read()
                    logger.debug("Loaded optimization plan from %s", path)
                    loaded_code.append(code)
            else:
                break
        else:
            logger.info("Loaded %d code implementations rather than generating new ones", num_samples)
            loaded_candidates = []
            for c_i in range(num_samples):
                loaded_candidates.append(CodeCandidate(candidates, "Combined code", loaded_code[c_i]), code_gen_model=self.llm_client.model)
            return loaded_candidates

        if self.pe_dim == 4:
            prompt_text = "The Gemmini accelerator's ISA is as follows:" + isa_prompt_admm.PROMPT(self.pe_dim)
        elif self.pe_dim == 16:
            prompt_text = "The Gemmini accelerator's ISA is as follows:" + isa_prompt_conv.PROMPT(self.pe_dim)
        prompt_text += "\nYou are an optimizing compiler generating high-performance Gemmini code. Let's combine the following optimized code samples to extract the high-performance characteristics of each:\n"
        for i, c in enumerate(candidates):
            prompt_text += f"Sample {i+1}:\n{c.code}\n"

        prompt_text += "\nMake sure to follow these rules:"
        prompt_text += gemmini_rules.PROMPT()
        prompt_text += "Optimized code:"

        # Save full prompt
        prompt_path = save_dir / f"prompt{'' if not save_str else '_' + save_str}.txt"
        with open(prompt_path, "w") as f:
            f.write(prompt_text)
        
        messages = [
            {"role": "user", "content": prompt_text},
        ]
        temperature = 1
        responses = self.llm_client.chat(
            messages=messages,
            num_candidates=num_samples,  # number of candidates,
            temperature=temperature
        )
        candidates = []
        for c_i, c in enumerate(responses):
            # Save full response
            full_path = save_dir / f"combined{'' if not save_str else '_' + save_str}_{c_i}_full.txt"
            with open(full_path, "w") as f:
                f.write(c)
            # Extract just the code
            extracted_code = extract(c)
            if not extracted_code:
                logger.warning("Failed to extract code from response %d, full response was %s", c_i, c)
            path = save_dir / f"combined{'' if not save_str else '_' + save_str}_{c_i}.txt"
            with open(path, "w") as f:
                f.writelines(extracted_code)
            logger.debug("Saved combined code %d to %s", c_i, path)

            candidates.append(CodeCandidate(candidates, "Combine parents", extracted_code, code_gen_model=self.llm_client.model))
        return candidates

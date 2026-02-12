import pathlib
import random

from autocomp.common import logger
from autocomp.search.prob import Prob
from autocomp.search.code_repo import CodeCandidate
from autocomp.agents.llm_agent import LLMAgent
from autocomp.agents.gemmini.prompts import isa_prompt_conv, isa_prompt_admm, tiling_example, if_example
from autocomp.hw_config.gemmini_config import GemminiHardwareConfig
from autocomp.backend.eval_backend import EvalBackend

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

class GemminiLLMAgent(LLMAgent):
    def __init__(self, model, hw_config: GemminiHardwareConfig, eval_backend: EvalBackend):
        super().__init__(model)
        self.hw_config = hw_config
        self.eval_backend = eval_backend
        self.pe_dim = hw_config.pe_dim

    def __repr__(self):
        return f"GemminiLLMAgent({self.llm_client.model}, {self.pe_dim})"

    def _get_prompt_rules(self, planning: bool, coding: bool) -> str:
        rules = []
        rules.extend(self.hw_config.get_hw_config_specific_rules())
        rules.extend(self.eval_backend.get_backend_specific_rules())
        rules.extend([
            "The rewritten program should be semantically equivalent to the original program",
            "If modifying loops, modify other related loop bounds and adjust address and index calculations to ensure the code is still correct",
            "If increasing loaded tile size, ensure that data is spread throughout the scratchpad across all relevant dimensions",
            "If loading across new dimensions, add the loop indices of those dimensions to scratchpad address calculations",
            "If increasing loaded tile size, update preload and compute instructions to match the new data layout",
            "If increasing loaded tile size, update base scratchpad addresses to fit new tile size",
        ])
        if planning:
            rules.append("Limit the scope of the plan to the selected optimization.")
        if coding:
            rules.append("Wrap the generated code with ```c at the beginning and ``` at the end.")
        rules_text = "\nRules:\n"
        for i, rule in enumerate(rules):
            rules_text += f"{i+1}. {rule}\n"
        return rules_text

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
        else:
            prompt_text = "The Gemmini accelerator's ISA is as follows:" + isa_prompt_conv.PROMPT(self.pe_dim) + "\n"
        for rule in self.hw_config.get_hw_config_specific_rules():
            prompt_text += rule + "\n"
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

    def _get_propose_optimizations_prompt(self, candidate: CodeCandidate,
                                          prob: Prob,
                                          force_opt_menu: int, 
                                          prompt_end: str, 
                                          analysis: str, 
                                          shuffle_opts: bool, 
                                          give_score_feedback: float,
                                          give_util_feedback: float,
                                          give_hw_feedback: float,
                                          include_ancestors: bool,
                                          plan_icl_examples: bool,
                                          cur_iter: int,
                                          num_iters: int,
                                          dropout_menu_options: float,
                                          translate: bool,
                                         ) -> str:
        # Select which menu options will appear
        menu_options_text = ""
        opt_lst = self.get_opt_menu_options(prob)
        if dropout_menu_options < 1 and not force_opt_menu:
            opt_lst = [opt for opt in opt_lst if random.random() < dropout_menu_options]
        if shuffle_opts:
            random.shuffle(opt_lst)
        include_util_feedback = random.random() < give_util_feedback
        include_score_feedback = random.random() < give_score_feedback
        include_hw_feedback = random.random() < give_hw_feedback

        parents_prompt = ""
        cur_cand = candidate
        while cur_cand is not None:
            # Go up to each parent and append to front of prompt
            # annotated_code = GemminiCode(cur_cand.code, self.pe_dim).annotate_perf()
            if include_hw_feedback:
                parents_prompt = "\n".join(cur_cand.hw_feedback) + "\n" + parents_prompt
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
        if self.pe_dim == 4:
            prompt_text = "\nThe Gemmini accelerator's ISA is as follows:" + isa_prompt_admm.PROMPT(self.pe_dim)
        else:
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
        prompt_text += """Please carefully review the program to identify any inefficiencies. 
Cycles can be reduced by using the following optimizations:
<optimizations>: \n""" + menu_options_text + "\n"

        if force_opt_menu:
            prompt_text += "Explain how to apply <optimization> " + str(force_opt_menu) + ": '" + opt_lst[force_opt_menu-1] + "' to the above code to reduce cycle count, and explain how it will improve performance."
        else:
            prompt_text += "You are an optimizing compiler that generates high-performance Gemmini code. Come up with a plan to apply exactly one of the <optimizations> to address the inefficiencies of the above code and reduce its cycle count. The plan should be specific to this code and explain how to change it."

        prompt_text += "\nMake sure to follow these rules:"
        prompt_text += self._get_prompt_rules(planning=True, coding=False)

        if prompt_end:
            logger.debug("Appended the following as prompt_end: '%s'", prompt_end)
            prompt_text += "\n" + prompt_end
        return prompt_text

    def _get_implement_code_prompt(self, candidate: CodeCandidate, prob: Prob = None, code_icl_examples: bool = True) -> str:
        if self.pe_dim == 4:
            prompt_text = "The Gemmini accelerator's ISA is as follows:" + isa_prompt_admm.PROMPT(self.pe_dim)
        else:
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
        prompt_text += self._get_prompt_rules(planning=False, coding=True)

        # prompt_text += "\nRespond with only the optimized code:"
        prompt_text += "Optimized code:"

        return prompt_text

    def _get_combine_candidates_prompt(self, candidates: list[CodeCandidate], prob: Prob = None) -> str:
        if self.pe_dim == 4:
            prompt_text = "The Gemmini accelerator's ISA is as follows:" + isa_prompt_admm.PROMPT(self.pe_dim)
        else:
            prompt_text = "The Gemmini accelerator's ISA is as follows:" + isa_prompt_conv.PROMPT(self.pe_dim)
        prompt_text += "\nYou are an optimizing compiler generating high-performance Gemmini code. Let's combine the following optimized code samples to extract the high-performance characteristics of each:\n"
        for i, c in enumerate(candidates):
            prompt_text += f"Sample {i+1}:\n{c.code}\n"

        prompt_text += "\nMake sure to follow these rules:"
        prompt_text += self._get_prompt_rules(planning=False, coding=True)
        prompt_text += "Optimized code:"
        return prompt_text

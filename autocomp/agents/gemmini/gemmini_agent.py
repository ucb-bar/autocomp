import pathlib
import random

from autocomp.common import logger
from autocomp.search.prob import Prob
from autocomp.search.code_repo import CodeCandidate
from autocomp.agents.llm_agent import LLMAgent
from autocomp.agents.gemmini.prompts import isa_prompt_conv, isa_prompt_admm, plan_prompt, gemmini_rules, tiling_example, if_example

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
                                          translate: bool,
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

    def _get_implement_code_prompt(self, candidate: CodeCandidate, prob: Prob = None, code_icl_examples: bool = True) -> list[CodeCandidate]:
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

    def _get_combine_candidates_prompt(self, candidates: list[CodeCandidate], prob: Prob = None) -> str:
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
        return prompt_text

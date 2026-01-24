import random

from autocomp.common import logger
from autocomp.agents.llm_agent import LLMAgent
from autocomp.search.prob import Prob
from autocomp.search.code_repo import CodeCandidate

class RVVLLMAgent(LLMAgent):
    def __init__(self, model):
        super().__init__(model)

    def get_opt_menu_options(self) -> list[str]:
        return [
                "modify loop tiling",
                "loop reordering",
                "split loops",
                "fuse loops",
                "simplify arithmetic and propagate constants to simplify expressions",
                "reorder computations or blocks of computations",
                "loop unrolling",
                "prefetching",
                "double buffering",
                "register blocking",
                "maximize LMUL",
                "use fused operations or instructions",
                "hoist redundant operations out of loops",
                "substitute operations with equivalent operations that are faster",
                "pipeline operations to better overlap computation and data movement",
                "minimize data movement",
                "minimize loop overhead",
                "other methods not listed here.",
        ]

    def _get_prompt_rules(self) -> str:
        return """
1. The rewritten program should be semantically equivalent to the original program.
2. Limit the scope of the plan to the selected optimization.
3. Only function gemm_f32 will be imported during evaluation. Feel free to define new variables, functions, etc., but make sure they are used by gemm_f32.
4. The code will be compiled with the following command:
```
riscv64-unknown-linux-gnu-gcc -S -O1 -march=rv64gcv -mabi=lp64d -mcmodel=medany code.c -o code.s
```
"""

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
        opt_lst = self.get_opt_menu_options()
        if dropout_menu_options < 1 and not force_opt_menu:
            opt_lst = [opt for opt in opt_lst if random.random() < dropout_menu_options]
        if shuffle_opts:
            random.shuffle(opt_lst)
        include_score_feedback = random.random() < give_score_feedback

        parents_prompt = ""
        cur_cand = candidate
        while cur_cand is not None:
            # Go up to each parent and append to front of prompt
            if include_score_feedback and (cur_cand.score is not None):
                parents_prompt = f"The latency of this code was {cur_cand.score} seconds.\n" + parents_prompt
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
        prompt_text = "This code is running on a Kendryte K230 with a XuanTie C908 RVV 1.0 compliant processor with 128-bit VLEN and 32KB L1 cache."
        prompt_text += parents_prompt

        # Now add the actual planning prompt
        menu_options_text = ""
        for i, opt in enumerate(opt_lst):
            menu_options_text += f"{i+1}. {opt}\n"
        prompt_text += """Please carefully review the program to identify any inefficiencies. 
Speedup can be increased by using the following optimizations:
<optimizations>: \n""" + menu_options_text + "\n"
        
        if force_opt_menu:
            prompt_text += "Explain how to apply <optimization> " + str(force_opt_menu) + ": '" + opt_lst[force_opt_menu-1] + "' to the above code to reduce execution time, and explain how it will improve performance."
        else:
            prompt_text += "You are a vector processor expert generating high-performance RVV code. Come up with a plan to apply exactly one of the <optimizations> to address the inefficiencies of the above code and reduce its execution time. The plan should be specific to this code and explain how to change it."

        prompt_text += "\nMake sure to follow these rules:"
        prompt_text += self._get_prompt_rules()

        if prompt_end:
            logger.debug("Appended the following as prompt_end: '%s'", prompt_end)
            prompt_text += "\n" + prompt_end
        return prompt_text


    def _get_implement_code_prompt(self, candidate: CodeCandidate, prob: Prob = None, code_icl_examples: bool = True) -> list[CodeCandidate]:
        prompt_text = "\nThe original code is as follows:\n```c\n"
        prompt_text += candidate.parent.code
        prompt_text += "\n```\nYou are a vector processor expert generating high-performance RVV code. Let's optimize the original code based on the following plan:\n"
        prompt_text += candidate.plan

        prompt_text += "\nMake sure to follow these rules:"
        prompt_text += self._get_prompt_rules()

        prompt_text += "Optimized code:"

        return prompt_text
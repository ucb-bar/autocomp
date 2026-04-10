import pathlib
import random

from autocomp.common import logger
from autocomp.search.prob import Prob
from autocomp.search.code_repo import CodeCandidate
from autocomp.agents.llm_agent import LLMAgent
from autocomp.agents.trn_nki2.prompts import fusion_example
from autocomp.agents.trn_nki2.nki_isa_generator import NkiIsaGenerator
from autocomp.hw_config.trn_config import TrnHardwareConfig
from autocomp.backend.eval_backend import EvalBackend


class TrnNki2LLMAgent(LLMAgent):
    def __init__(self, model, hw_config: TrnHardwareConfig, eval_backend: EvalBackend):
        super().__init__(model)
        self.hw_config = hw_config
        self.eval_backend = eval_backend
        self.nki_isa_generator = NkiIsaGenerator()
        self._isa_selection_cache: dict[str, str] = {}

    def __repr__(self):
        return f"TrnNki2LLMAgent({self.llm_client.model})"

    def _match_names_from_llm(self, prompt: str, valid_names: list[str]) -> list[str]:
        """Send a selection prompt to the LLM and fuzzy-match the response to valid names."""
        try:
            responses = self.llm_client.chat(prompt=prompt, num_candidates=1, temperature=0)
            raw = responses[0] if responses else ""
            selected: list[str] = []
            for line in raw.strip().split("\n"):
                line = line.strip().lstrip("- ").strip()
                if not line:
                    continue
                if line in valid_names:
                    if line not in selected:
                        selected.append(line)
                else:
                    for name in valid_names:
                        if line.lower() in name.lower() or name.lower() in line.lower():
                            if name not in selected:
                                selected.append(name)
                            break
            return selected if selected else valid_names
        except Exception:
            logger.warning("LLM ISA selection failed, falling back to all instructions")
            return valid_names

    def _select_isa_instructions(self, prob: Prob, code: str) -> list[str]:
        """Use LLM to select which ISA instructions are relevant for the given code."""
        all_names = self.nki_isa_generator.get_all_instruction_names()
        standard_names = self.nki_isa_generator.get_standard_instruction_names()
        non_standard_names = [n for n in all_names if n not in standard_names]

        summary_parts = []
        for name in non_standard_names:
            summary = self.nki_isa_generator.get_instruction_summary(name)
            if summary:
                summary_parts.append(f"- {name}: {summary}")
            else:
                summary_parts.append(f"- {name}")
        summary_list = "\n".join(summary_parts)

        prob_context = getattr(prob, "context", "")
        prompt = (
            "Given the following NKI kernel code and problem context, select which ISA "
            "instructions/APIs are relevant for optimizing this code. Return ONLY the "
            "instruction names, one per line.\n\n"
            f"Problem type: {prob.prob_type}\n"
            f"{f'Problem context: {prob_context}' if prob_context else ''}\n\n"
            f"Code:\n```\n{code}\n```\n\n"
            f"Available ISA instructions:\n{summary_list}\n\n"
            "Relevant instructions (one per line):"
        )
        selected = self._match_names_from_llm(prompt, non_standard_names)
        return standard_names + selected

    def _get_isa_for_problem(self, prob: Prob, code: str) -> str:
        """Get ISA documentation for a problem, using LLM selection with caching."""
        cache_key = f"{prob.prob_type}:{prob.prob_id}"
        if cache_key in self._isa_selection_cache:
            return self._isa_selection_cache[cache_key]

        selected_names = self._select_isa_instructions(prob, code)
        logger.debug("LLM selected %d ISA instructions for %s", len(selected_names), cache_key)
        isa_text = self.nki_isa_generator.generate_isa_from_names(selected_names)
        self._isa_selection_cache[cache_key] = isa_text
        return isa_text

    def _get_convert_to_nki_menu_options(self) -> list[str]:
        return [
            "convert non-NKI code into NKI code",
            "move non-NKI code into a NKI kernel",
            "move a non-NKI transpose into a NKI kernel",
            "fuse multiple NKI kernels into a single kernel",
        ]

    def get_opt_menu_options(self, prob: Prob):
        """Get optimization menu options for NKI/Trainium kernels"""
        return [
            "eliminate loads and stores as much as possible, keeping data in SBUF/PSUM instead",
            "minimize data movement",
            "overlap data movement and compute",
            "improve data layout and access patterns",
            "loop reordering and restructuring",
            "Load larger blocks of data to increase SBUF data reuse and reduce memory traffic",
            "Add additional loop levels so larger blocks of data can be loaded",
            "avoid rematerializing",
            "skip computation when it is not needed (e.g. it is completely masked out)",
            "fuse loops (reordering if necessary)",
            "increase reuse by keeping data in SBUF across outer loop iterations",
            "hoist redundant operations out of loops",
            "delay softmax division until after all reductions are complete",
            # "Perform nc_matmul on large contiguous blocks within its own affine_range loop to maximize compute throughput.",
            # "Group nc_matmul calls into larger blocks, organizing inputs ahead of time, to maximize Tensor Engine utilization.",
            "do operations in lower precision such as nl.bfloat16",
            "double buffering",
            "loop unrolling",
            "Multi-tile grouping",
            "Supertile fuse-and-reuse for loads",
            "fuse multiple instructions into one, for example by doing reduction inside nisa.activation()",
            "software pipelining",
            "keep data in SBUF/PSUM instead of storing to and loading from HBM",
            "stronger tiling for contraction / moving-free split",
            "reorder operations to improve locality",
            "fuse dependent operations",
            "fuse operations into a single loop so intermediate data does not need to be stored to and loaded from HBM",
            "fuse loops that iterate over the same dimension to improve intermediate data reuse",
            "allocate a larger tile in SBUF so we can keep data in it rather than storing to and loading from HBM",
            "allocate buffers in lower precision such as nl.bfloat16",
            "downcast to lower precision during operations that take dtype as an argument",
            "keep data in the same layout to avoid transpose operations",
            "eliminate intermediate tensor materialization by using in-place operations (storing the output in the same buffer as the input)",
            "use the streaming softmax with running max and scaling trick",
            "optimize accumulation patterns in PSUM",
            "optimize reduction by fusing tile-wise reductions with transformation passes",
            "Combine adjacent tiles into contiguous blocks before nisa.dma_copy() to maximize memory throughput.",
            "Scan carry-over to parallelize the scan operation",
            "Hoist nisa.dma_copy() operations for reused data (e.g., LHS tiles) outside inner loops to reduce redundant HBM→SBUF transfers.",
            "Kernel Fusion via SBUF residency",
            "Add a small fixed-size inner loop as a compiler hint",
            "Modify one particular parameter",
            "Target the specific data shapes and shapes of the input and output tensors",
            "Tile Vector Engine instructions in loops of size 128 to coalesce them",
            "Use a different engine for an operation",
            "inline a function so it can be more easily optimized and fused",
            # "Replace general-purpose code with faster specialized instructions",
            # "transpose inside the NKI kernel",
            # "move non-NKI code into the NKI kernel",
            "Overlap execution across compute engines through pipelining",
            "Swap stationary and moving tensors in nc_matmul",
            "Use conditional execution instead of masking, or vice versa",
            "Simplify or eliminate any unnecessary code",
            "Other methods not listed here.",
        ]

    def _get_prompt_rules(self, planning: bool, coding: bool, prob: Prob = None) -> str:
        rules = []
        rules.extend(self.hw_config.get_hw_config_specific_rules())
        rules.extend(self.eval_backend.get_backend_specific_rules())
        rules.extend([
                 "The rewritten program should be semantically equivalent to the original program, within a small numerical tolerance.",
                 "Keep the same function name and signature as the original program (helper functions can be renamed or deleted).",
                 "Maintain correct tensor shapes and indexing patterns. Remember not to index with affine_range loop variables. Avoid loop carried dependencies.",
                 "The following imports have already been run: import nki; import nki.isa as nisa; import nki.language as nl; import numpy as np;",
                ])
        if planning:
            rules.append("Limit the scope of the plan to the selected optimization.")
            if random.random() < 0.4:
                rules.append("Limit the scope of the plan so that the rewritten program is still correct.")
            elif random.random() < 0.3:
                rules.append("Plans can be highly targeted to one particular part of the code.")
            rules.append("Do not count out any of the <optimizations> unless they are clearly irrelevant to the code.")
        if coding:
            rules.append("Optimize the solution() function and do not change its name.")
            rules.append("Wrap the generated code with ```python at the beginning and ``` at the end.")
        rules.append("Ensure that loop dependencies are not violated inside affine_range loops.")

        prompt_text = ""
        for i, rule in enumerate(rules):
            prompt_text += f"{i+1}. {rule}\n"
        return prompt_text

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
        if translate:
            opt_lst = self._get_convert_to_nki_menu_options()
        else:
            opt_lst = self.get_opt_menu_options(prob)
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
                parents_prompt = f"The latency of this code was {cur_cand.score} ms.\n" + parents_prompt
            if not include_ancestors:
                parents_prompt = "\nThe original unoptimized code was:\n```\n" + cur_cand.code + "\n```\n" + parents_prompt
                break # No need to go up past the immediate parent
            elif cur_cand.plan is not None:
                parents_prompt = "\nNext, we applied this plan to the code:\n" + cur_cand.plan + "\nThe generated code was:\n" + cur_cand.code + "\n" + parents_prompt
            else:
                parents_prompt = "\nThe original unoptimized code was:\n```\n" + cur_cand.code + "\n```\n" + parents_prompt
            cur_cand = cur_cand.parent

        if analysis:
            parents_prompt += "\n" + analysis

        # Initialize the prompt with NKI context
        prompt_text = "The NKI (Neuron Kernel Interface) is used for writing high-performance kernels on AWS Trainium and Inferentia chips.\n"
        prompt_text += self._get_isa_for_problem(prob, candidate.code)
        
        prompt_text += parents_prompt

        # Now add the actual planning prompt
        for i, opt in enumerate(opt_lst):
            menu_options_text += f"{i+1}. {opt}\n"
        
        prompt_text += "Please carefully review the NKI code to identify any inefficiencies. "
        prompt_text += "Performance can be improved by using the following optimizations:\n"
        prompt_text += "<optimizations>:\n" + menu_options_text + "\n"
        
        if force_opt_menu:
            prompt_text += "Explain how to apply <optimization> " + str(force_opt_menu) + ": '" + opt_lst[force_opt_menu-1] + "' to the above code to reduce execution time, and explain how it will improve performance."
        else:
            prompt_text += "You are an expert NKI performance engineer generating high-performance Trainium/Inferentia kernels. "

            # prompt_text += "Come up with a plan to apply exactly one of the <optimizations> to address the inefficiencies of the above code and reduce its execution time. The plan should be specific to this code and explain how to change it."
            # TODO make it a parameter
            choose_or_invent = random.random()
            if choose_or_invent < 0.1 and not translate:
                # Prompt to invent a new optimization inspired by the <optimizations>
                prompt_text += "Invent a new optimization inspired by the <optimizations> to apply to the above code to reduce execution time, and explain how it will improve performance."
            elif choose_or_invent < 0.2 and not translate:
                # Prompt to invent a new optimization different from the <optimizations>
                prompt_text += "Think of a new optimization different from the <optimizations> to apply to the above code to reduce execution time, and explain how it will improve performance."
            else:
                prompt_text += "Come up with a plan to apply exactly one of the <optimizations> to address the inefficiencies of the above code and reduce its execution time."

        prompt_text += " The plan should be specific to this code and explain how to change it."
        # # TODO make it a parameter
        # if random.random() < 0.5:
        #     prompt_text += " The plan should be specific to this code and explain how to change it."
        prompt_text += "\nMake sure to follow these rules:\n"
        prompt_text += self._get_prompt_rules(planning=True, coding=False, prob=prob)

        if prompt_end:
            logger.debug("Appended the following as prompt_end: '%s'", prompt_end)
            prompt_text += "\n" + prompt_end
        return prompt_text

    def _get_implement_code_prompt(self, candidate: CodeCandidate, prob: Prob = None, code_icl_examples: bool = True) -> str:
        prompt_text = "The NKI (Neuron Kernel Interface) is used for writing high-performance kernels on AWS Trainium and Inferentia chips.\n"
        if prob is None:
            raise ValueError("TrnNki2LLMAgent requires prob parameter to be provided")
        prompt_text += self._get_isa_for_problem(prob, candidate.parent.code)

        if "fusion" in candidate.plan.lower() or "fuse" in candidate.plan.lower():
            rand_val = random.random()
            if rand_val < 0.1:
                prompt_text += "\n" + fusion_example.PROMPT() + "\n"
            elif rand_val < 0.2:
                prompt_text += "\n" + fusion_example.PROMPT_2() + "\n"
            elif rand_val < 0.3:
                prompt_text += "\n" + fusion_example.PROMPT_3() + "\n"

        prompt_text += "The original code is as follows:\n"
        prompt_text += candidate.parent.code
        prompt_text += "\nYou are an expert NKI performance engineer generating high-performance Trainium/Inferentia kernels. "
        prompt_text += "Let's optimize the original code based on the following plan:\n"
        prompt_text += candidate.plan

        prompt_text += "\nMake sure to follow these rules:\n"
        prompt_text += self._get_prompt_rules(planning=False, coding=True, prob=prob)
        prompt_text += "\nOptimized NKI code:"

        return prompt_text

    def _get_direct_implement_prompt(self, candidate: CodeCandidate, prob: Prob,
                                     give_score_feedback: float = 1.0,
                                     give_hw_feedback: float = 1.0,
                                     include_ancestors: bool = False,
                                     dropout_menu_options: float = 1.0,
                                     cur_iter: int = None,
                                     num_iters: int = None,
                                     translate: bool = False) -> str:
        if translate:
            opt_lst = self._get_convert_to_nki_menu_options()
        else:
            opt_lst = self.get_opt_menu_options(prob)
            if dropout_menu_options < 1:
                opt_lst = [opt for opt in opt_lst if random.random() < dropout_menu_options]

        include_score_feedback = random.random() < give_score_feedback

        parents_prompt = ""
        cur_cand = candidate
        while cur_cand is not None:
            if include_score_feedback and (cur_cand.score is not None):
                parents_prompt = f"The latency of this code was {cur_cand.score} ms.\n" + parents_prompt
            if not include_ancestors:
                parents_prompt = "\nThe original unoptimized code was:\n```\n" + cur_cand.code + "\n```\n" + parents_prompt
                break
            elif cur_cand.plan is not None:
                parents_prompt = "\nNext, we applied this plan to the code:\n" + cur_cand.plan + "\nThe generated code was:\n" + cur_cand.code + "\n" + parents_prompt
            else:
                parents_prompt = "\nThe original unoptimized code was:\n```\n" + cur_cand.code + "\n```\n" + parents_prompt
            cur_cand = cur_cand.parent

        prompt_text = "The NKI (Neuron Kernel Interface) is used for writing high-performance kernels on AWS Trainium and Inferentia chips.\n"
        prompt_text += self._get_isa_for_problem(prob, candidate.code)
        prompt_text += parents_prompt

        menu_options_text = ""
        for i, opt in enumerate(opt_lst):
            menu_options_text += f"{i + 1}. {opt}\n"

        if translate:
            prompt_text += "Please review the code and identify parts that should be converted to NKI kernels.\n"
            prompt_text += "The following conversion strategies are available:\n"
            prompt_text += "<optimizations>:\n" + menu_options_text + "\n"
            prompt_text += "You are an expert NKI performance engineer. "
            prompt_text += "Apply one of the <optimizations> to convert the above code to NKI. "
            prompt_text += "First, plan your approach, then output the complete code.\n"
        else:
            prompt_text += "Please carefully review the NKI code to identify any inefficiencies. "
            prompt_text += "Performance can be improved by using the following optimizations:\n"
            prompt_text += "<optimizations>:\n" + menu_options_text + "\n"
            prompt_text += "You are an expert NKI performance engineer generating high-performance Trainium/Inferentia kernels. "
            prompt_text += "Apply one of the <optimizations> to address the inefficiencies of the above code and reduce its execution time. "
            prompt_text += "First, plan your approach, then output the complete optimized code.\n"

        prompt_text += "\nMake sure to follow these rules:\n"
        prompt_text += self._get_prompt_rules(planning=True, coding=True, prob=prob)

        if cur_iter is not None and num_iters is not None:
            prompt_text += f"\nRemember that this is phase {cur_iter} out of {num_iters} optimization phases."

        prompt_text += "\nOptimized NKI code:"
        return prompt_text

    def _get_combine_candidates_prompt(self, candidates: list[CodeCandidate], prob: Prob = None) -> str:
        prompt_text = "The NKI (Neuron Kernel Interface) is used for writing high-performance kernels on AWS Trainium and Inferentia chips.\n"
        prompt_text += "You are an expert NKI performance engineer generating high-performance Trainium/Inferentia kernels. "
        prompt_text += "Let's combine the following optimized NKI code samples to extract the high-performance characteristics of each:\n"
        for i, c in enumerate(candidates):
            prompt_text += f"Sample {i+1}:\n{c.code}\n"

        prompt_text += "\nMake sure to follow these rules:"
        prompt_text += self._get_prompt_rules(planning=False, coding=True, prob=prob)
        prompt_text += "\nOptimized combined NKI code:"
        return prompt_text

    def _get_reimplement_failed_code_prompt(self, candidate: CodeCandidate, prob: Prob = None) -> str:
        """
        Generate a prompt to reimplement failed code based on stdout/stderr feedback.
        """
        if prob is None:
            raise ValueError("TrnNki2LLMAgent requires prob parameter to be provided")

        prompt_text = "The NKI (Neuron Kernel Interface) is used for writing high-performance kernels on AWS Trainium and Inferentia chips.\n"
        prompt_text += self._get_isa_for_problem(prob, candidate.code)

        # prompt_text += "The original code is as follows:\n"
        # prompt_text += candidate.parent.code
        prompt_text += "\n\nYou are an expert NKI performance engineer generating high-performance Trainium/Inferentia kernels. "
        # prompt_text += "We attempted to optimize the original code based on the following plan:\n"
        # prompt_text += candidate.plan
        # prompt_text += "\n\nThe generated code was:\n"
        prompt_text += "\nThe code was:\n"
        prompt_text += candidate.code
        
        # Add error information
        prompt_text += "\n\nHowever, the code failed with the following output:\n"
        if candidate.stderr:
            prompt_text += "=== STDERR ===\n"
            # Limit the length of each line to 400 characters
            stderr_lines = candidate.stderr.split("\n")
            stderr_lines = [line[:400] for line in stderr_lines]
            stderr_lines = "\n".join(stderr_lines)
            prompt_text += stderr_lines
            prompt_text += "\n"
        if candidate.stdout:
            prompt_text += "=== STDOUT ===\n"
            stdout_lines = candidate.stdout.split("\n")
            stdout_lines = [line[:400] for line in stdout_lines]
            stdout_lines = "\n".join(stdout_lines)
            prompt_text += stdout_lines
            prompt_text += "\n"
        
        prompt_text += "\nPlease fix the code to address the errors while still applying the optimization plan. "
        prompt_text += "Make sure to follow these rules:\n"
        prompt_text += self._get_prompt_rules(planning=False, coding=True, prob=prob)
        prompt_text += "\nFixed and optimized NKI code:"

        return prompt_text

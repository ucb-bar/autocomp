import pathlib
import random

from autocomp.common import logger
from autocomp.search.prob import Prob
from autocomp.search.code_repo import CodeCandidate
from autocomp.agents.llm_agent import LLMAgent


class TpuLLMAgent(LLMAgent):
    def __init__(self, model):
        super().__init__(model)

    def __repr__(self):
        return f"TpuLLMAgent({self.llm_client.model})"

    def propose_optimizations(
        self,
        candidate: CodeCandidate,
        num_plans: int,
        save_dir: pathlib.Path,
        save_str: str,
        prob: Prob,
        force_opt_menu: int = None,
        prompt_end: str = "",
        analysis: str = "",
        shuffle_opts: bool = False,
        give_score_feedback: float = 1.0,
        give_util_feedback: float = 0.0,
        give_spad_acc_feedback: float = 1.0,
        include_ancestors: bool = True,
        plan_icl_examples: bool = False,
        cur_iter: int = None,
        num_iters: int = None,
        dropout_menu_options: float = 1,
        translate: bool = False,
    ) -> list[CodeCandidate]:
        """
        TPU prompts expect a hardware-feedback knob; route through the parallel
        planner so we can pass it (and keep `LLMAgent` close to upstream).
        """
        force_opt_menu_lst = None if force_opt_menu is None else [force_opt_menu]
        analysis_lst = None if analysis is None else [analysis]
        return super().propose_optimizations_parallel(
            candidate_lst=[candidate],
            num_plans=num_plans,
            save_dir=save_dir,
            save_strs=[save_str],
            prob=prob,
            force_opt_menu_lst=force_opt_menu_lst,
            prompt_end=prompt_end,
            analysis_lst=analysis_lst,
            shuffle_opts=shuffle_opts,
            give_score_feedback=give_score_feedback,
            give_util_feedback=give_util_feedback,
            give_hw_feedback=give_spad_acc_feedback,
            include_ancestors=include_ancestors,
            plan_icl_examples=plan_icl_examples,
            cur_iter=cur_iter,
            num_iters=num_iters,
            dropout_menu_options=dropout_menu_options,
            translate=translate,
        )

    def _get_convert_to_pallas_menu_options(self) -> list[str]:
        return [
            "convert non-Pallas code into Pallas kernel code",
            "move non-Pallas code into a Pallas kernel",
            "fuse multiple Pallas kernels into a single kernel",
        ]

    def get_opt_menu_options(self, prob: Prob):
        """Get optimization menu options for Pallas/TPU kernels"""
        return [
            "eliminate loads and stores as much as possible, keeping data in VMEM instead",
            "minimize data movement between HBM and VMEM",
            "overlap data movement and compute using pipelining",
            "improve data layout and access patterns in VMEM",
            "loop reordering and restructuring",
            "avoid rematerializing intermediate tensors",
            "inline a function so it can be more easily optimized and fused",
            "skip computation when it is not needed (e.g. it is completely masked out)",
            "fuse loops (reordering if necessary)",
            "increase reuse by keeping data in VMEM across outer loop iterations",
            "hoist redundant operations out of loops",
            "delay softmax division until after all reductions are complete",
            "Perform matrix operations on large contiguous blocks within their own loop to maximize compute throughput.",
            "Group matrix operations into larger blocks, organizing inputs ahead of time, to maximize TPU utilization.",
            "do operations in lower precision such as bfloat16",
            "double buffering",
            "fuse multiple operations into one kernel",
            "software pipelining",
            "keep data in VMEM instead of storing to and loading from HBM",
            "stronger tiling for contraction / moving-free split",
            "reorder operations to improve locality",
            "fuse dependent operations",
            "fuse operations into a single loop so intermediate data does not need to be stored to and loaded from HBM",
            "fuse loops that iterate over the same dimension to improve intermediate data reuse",
            "allocate a larger tile in VMEM so we can keep data in it rather than storing to and loading from HBM",
            "allocate buffers in lower precision such as bfloat16",
            "downcast to lower precision during operations",
            "keep data in the same layout to avoid transpose operations",
            "eliminate intermediate tensor materialization by using in-place operations (storing the output in the same buffer as the input)",
            "use the streaming softmax with running max and scaling trick",
            "optimize accumulation patterns in VMEM",
            "optimize reduction by fusing tile-wise reductions with transformation passes",
            "Load larger blocks of data to increase VMEM data reuse and reduce memory traffic",
            "Add additional loop levels so larger blocks of data can be loaded (multi-level tiling)",
            "Combine adjacent tiles into contiguous blocks before storing to maximize memory throughput.",
            "Scan carry-over to parallelize the scan operation",
            "Hoist memory load operations for reused data (e.g., LHS tiles) outside inner loops to reduce redundant HBM→VMEM transfers.",
            "Kernel Fusion via VMEM residency",
            "Modify one particular parameter to maximize performance",
            "Target the specific data shapes and shapes of the input and output tensors to maximize performance",
            "Tile operations in loops to coalesce them",
            "Use a different TPU operation for better performance",
            "Overlap execution across compute units through pipelining",
            "Swap stationary and moving tensors in matrix operations",
            "Use conditional execution instead of masking, or vice versa",
            "Simplify or eliminate any unnecessary code",
            "Other methods not listed here.",
        ]

    def _get_pallas_isa_documentation(self, prob: Prob = None) -> str:
        """Generate Pallas/TPU ISA documentation for prompts"""
        doc = """Pallas is JAX's kernel programming interface for TPUs. It allows writing custom kernels that run directly on TPU hardware.

Key concepts:
- VMEM (Vector Memory): On-chip memory for storing data during kernel execution
- HBM (High Bandwidth Memory): Off-chip memory accessed via loads/stores
- Kernel structure: Each Pallas kernel function receives memory references (x_ref, y_ref, o_ref) and operates on them
- Use pl.pallas_call() to wrap kernel functions and make them callable from JAX

Basic kernel pattern:
```python
def my_kernel(x_ref, y_ref, o_ref):
    # Read from VMEM
    x = x_ref[...]
    y = y_ref[...]
    # Compute
    result = x + y  # or other operations
    # Write to VMEM
    o_ref[...] = result

@jax.jit
def pallas_function(x, y):
    return pl.pallas_call(
        my_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
    )(x, y)
```

Memory operations:
- x_ref[...] reads entire tensor from VMEM
- o_ref[...] = value writes entire tensor to VMEM
- Can use slicing: x_ref[0:128, 0:64] for partial reads/writes

Important constraints:
- VMEM size is limited (typically 128KB per core)
- Operations should be tiled to fit in VMEM
- Use jax.block_until_ready() after kernel calls to ensure completion before timing
- Kernel functions should be pure (no side effects except through refs)

Performance tips:
- Minimize HBM↔VMEM transfers
- Maximize data reuse in VMEM
- Use appropriate tile sizes for your operation
- Consider precision (float32 vs bfloat16) for better performance
"""
        return doc

    def _get_prompt_rules(self, planning: bool, coding: bool, prob: Prob = None) -> str:
        rules = [
            "The rewritten program should be semantically equivalent to the original program, within a small numerical tolerance.",
            "Maintain correct tensor shapes and indexing patterns.",
            "The following imports have already been run: import jax; import jax.numpy as jnp; from jax.experimental import pallas as pl; from jax.experimental.pallas import tpu as pltpu; import numpy as np;",
            "Use pl.pallas_call() to wrap kernel functions and make them callable from JAX.",
            "Kernel functions should receive memory references (x_ref, y_ref, o_ref, etc.) and operate on them.",
            "Use jax.block_until_ready() after kernel calls to ensure completion before timing.",
        ]
        if planning:
            rules.append("Limit the scope of the plan to the selected optimization.")
            if random.random() < 0.4:
                rules.append("Limit the scope of the plan so that the rewritten program is still correct.")
            elif random.random() < 0.3:
                rules.append("Plans can be highly targeted to one particular part of the code.")
            rules.append("Do not count out any of the <optimizations> unless they are clearly irrelevant to the code.")
        if coding:
            rules.append("Optimize the test() function and do not change its name.")
            rules.append("Wrap the generated code with ```python at the beginning and ``` at the end.")
        rules.append("Ensure that loop dependencies are not violated.")

        # Problem-specific rules can be added here if needed
        # if prob and prob.prob_type == "tpu-tutorial" and prob.prob_id == 0:
        #     rules.append("You are optimizing for constant shapes: x.shape = (M, K), y.shape = (K, N). Make sure to take advantage of these shapes.")

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
                                          give_spad_acc_feedback: float,
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
            opt_lst = self._get_convert_to_pallas_menu_options()
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

        # Initialize the prompt with Pallas context
        prompt_text = "Pallas is JAX's kernel programming interface for TPUs, used for writing high-performance kernels on Google Cloud TPUs.\n"
        prompt_text += self._get_pallas_isa_documentation(prob)
        
        prompt_text += parents_prompt

        # Now add the actual planning prompt
        for i, opt in enumerate(opt_lst):
            menu_options_text += f"{i+1}. {opt}\n"
        
        prompt_text += "Please carefully review the Pallas code to identify any inefficiencies. "
        prompt_text += "Performance can be improved by using the following optimizations:\n"
        prompt_text += "<optimizations>:\n" + menu_options_text + "\n"
        
        if force_opt_menu:
            prompt_text += "Explain how to apply <optimization> " + str(force_opt_menu) + ": '" + opt_lst[force_opt_menu-1] + "' to the above code to reduce execution time, and explain how it will improve performance."
        else:
            prompt_text += "You are an expert Pallas/TPU performance engineer generating high-performance TPU kernels. "

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
        prompt_text += "\nMake sure to follow these rules:\n"
        prompt_text += self._get_prompt_rules(planning=True, coding=False, prob=prob)

        if prompt_end:
            logger.debug("Appended the following as prompt_end: '%s'", prompt_end)
            prompt_text += "\n" + prompt_end
        return prompt_text

    def _get_implement_code_prompt(self, candidate: CodeCandidate, prob: Prob = None, code_icl_examples: bool = True) -> str:
        prompt_text = "Pallas is JAX's kernel programming interface for TPUs, used for writing high-performance kernels on Google Cloud TPUs.\n"
        if prob is None:
            raise ValueError("TpuLLMAgent requires prob parameter to be provided")
        prompt_text += self._get_pallas_isa_documentation(prob)

        prompt_text += "The original code is as follows:\n"
        prompt_text += candidate.parent.code
        prompt_text += "\nYou are an expert Pallas/TPU performance engineer generating high-performance TPU kernels. "
        prompt_text += "Let's optimize the original code based on the following plan:\n"
        prompt_text += candidate.plan

        prompt_text += "\nMake sure to follow these rules:\n"
        prompt_text += self._get_prompt_rules(planning=False, coding=True, prob=prob)
        prompt_text += "\nOptimized Pallas code:"

        return prompt_text

    def _get_combine_candidates_prompt(self, candidates: list[CodeCandidate], prob: Prob = None) -> str:
        prompt_text = "Pallas is JAX's kernel programming interface for TPUs, used for writing high-performance kernels on Google Cloud TPUs.\n"
        prompt_text += "You are an expert Pallas/TPU performance engineer generating high-performance TPU kernels. "
        prompt_text += "Let's combine the following optimized Pallas code samples to extract the high-performance characteristics of each:\n"
        for i, c in enumerate(candidates):
            prompt_text += f"Sample {i+1}:\n{c.code}\n"

        prompt_text += "\nMake sure to follow these rules:"
        prompt_text += self._get_prompt_rules(planning=False, coding=True, prob=prob)
        prompt_text += "\nOptimized combined Pallas code:"
        return prompt_text

    def _get_reimplement_failed_code_prompt(self, candidate: CodeCandidate, prob: Prob = None) -> str:
        """
        Generate a prompt to reimplement failed code based on stdout/stderr feedback.
        """
        if prob is None:
            raise ValueError("TpuLLMAgent requires prob parameter to be provided")

        prompt_text = "Pallas is JAX's kernel programming interface for TPUs, used for writing high-performance kernels on Google Cloud TPUs.\n"
        prompt_text += self._get_pallas_isa_documentation(prob)

        prompt_text += "\n\nYou are an expert Pallas/TPU performance engineer generating high-performance TPU kernels. "
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
        prompt_text += "\nFixed and optimized Pallas code:"

        return prompt_text

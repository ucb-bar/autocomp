import pathlib
import random
import re

from autocomp.common import logger
from autocomp.search.prob import Prob
from autocomp.search.code_repo import CodeCandidate
from autocomp.agents.llm_agent import LLMAgent
from autocomp.backend.eval_backend import EvalBackend
from autocomp.hw_config.hardware_config import HardwareConfig
from autocomp.agents.tpu.prompts.isa_documentation import get_pallas_isa_documentation


class TpuLLMAgent(LLMAgent):
    def __init__(self, model: str, hw_config: HardwareConfig | None = None, eval_backend: EvalBackend | None = None):
        super().__init__(model)
        self.hw_config = hw_config
        self.eval_backend = eval_backend

    def __repr__(self):
        return f"TpuLLMAgent({self.llm_client.model})"

    def _get_convert_to_pallas_menu_options(self) -> list[str]:
        return [
            "convert non-Pallas code into Pallas kernel code",
            "move non-Pallas code into a Pallas kernel",
            "fuse multiple Pallas kernels into a single kernel",
        ]

    def get_opt_menu_options(self, prob: Prob):
        """Get optimization menu options for Pallas/TPU kernels"""
        base_opts = [
            # Memory movement / VMEM reuse
            "minimize data movement",
            "avoid rematerializing intermediate tensors",

            # Kernel and loop fusion
            "fuse multiple operations into one kernel",
            "fuse loops (reordering if necessary)",
            "inline a function so it can be more easily optimized and fused",
            "optimize reduction by fusing tile-wise reductions with transformation passes",

            # Layout and loop structure
            "improve data layout and access patterns",
            "loop reordering and restructuring",

            # Tiling and blocking
            "Add additional loop levels so larger blocks of data can be loaded (multi-level tiling)",
            "Group matrix operations into larger blocks, organizing inputs ahead of time",
            "Swap stationary and moving tensors in matrix operations",
            "Combine adjacent tiles into contiguous blocks before storing to maximize memory throughput",
            "Change block sizes",
            "Use asymmetric block sizes",
            "Maximize the reduction dimension block size",

            # Pipelining and overlap
            "overlap data movement and compute",
            "double buffering",
            "software pipelining",
            "Use PrefetchScalarGridSpec with grid_spec= instead of separate in_specs/out_specs/grid to enable automatic HBM-VMEM pipelining",

            # Loop-level computation optimization
            "hoist redundant operations out of loops",
            "skip computation when it is not needed (e.g. it is completely masked out)",

            # Reduction / attention-specific optimization
            "delay softmax division until after all reductions are complete",
            "use the streaming softmax with running max and scaling trick",
            "optimize accumulation patterns in VMEM",
            "Use a scratch VMEM buffer as the accumulator (scratch_shapes=[pltpu.VMEM(...)]) instead of accumulating directly into the output reference",

            # Precision optimization
            "do operations in lower precision such as bfloat16",

            # Hardware-specific operator choice
            "Use a different TPU operation for better performance",

            # Shape specialization
            "Target the specific data shapes and shapes of the input and output tensors to maximize performance",

            # General cleanup
            "Simplify or eliminate any unnecessary code",

            "Other methods not listed here.",
        ]
        opts = list(base_opts)
        if prob is not None and getattr(prob, "prob_type", None) == "tpu" and getattr(prob, "prob_id", None) == 0:
            opts.extend(
                [
                    "Replace hand-written Pallas matmul with jax.jit(jnp.matmul) (or jnp.matmul) so XLA lowers directly to the MXU—often faster than custom Pallas for dense GEMM.",
                    "Wrap the matmul in jax.jit with static argnums/keyword patterns so compilation happens once; keep test(x,y) as a thin call into the jitted function.",
                    "Use bfloat16 inputs with jnp.matmul under jax.jit when the benchmark dtype is bf16; match output dtype to inputs for harness correctness.",
                    "Cast inputs to bfloat16 before the matmul while keeping the accumulator in float32. The MXU processes bf16 at 2x the throughput of fp32. Use jax.lax.dot with preferred_element_type=jnp.float32 inside the kernel, or cast x_ref/y_ref to bfloat16 before the @ operator.",
                    "Use jax.lax.dot or jax.lax.dot_general instead of the @ operator inside the kernel to enable explicit transpose fusion and precision control via preferred_element_type.",
                    "Use pltpu.emit_pipeline for the inner K-reduction loop inside an outer pallas_call that partitions M/N tiles across megacore TensorCores. The outer kernel receives full HBM refs (memory_space=pl.ANY) and emit_pipeline manages all HBM->VMEM pipelining automatically.",
                    "Tune tile sizes (bm, bn, bk) jointly for the exact benchmark shapes. Maximize bm*bn for MXU reuse while ensuring m_tiles*n_tiles >= 2*num_cores for megacore utilization. Keep bk small (128-256) for pipelining depth.",
                    "Use pl.Buffered with use_lookahead=True on input BlockSpecs so the compiler begins prefetching tiles as soon as a buffer slot is free, rather than waiting until the iteration before they are needed.",
                    "On the first K iteration (kk==0) write the matmul result directly to z_ref instead of zeroing then accumulating. This eliminates a full-tile zero-store and avoids an unnecessary output-tile prefetch on the first iteration.",
                    "Reorder the grid axes or swap which operand is stationary vs streaming to improve data reuse. For example, making the LHS (x) stationary across consecutive j iterations can reduce redundant HBM->VMEM transfers.",
                    "Use scratch_shapes to allocate explicit VMEM scratch buffers for manual double-buffering or for staging intermediate results, giving more control than compiler-managed pipelining alone.",
                    "Increase parallel work by ensuring the grid has enough output tiles for both TensorCores. If bm and bn are too large, the grid may have too few tiles to keep both cores busy.",
                    "Simplify or remove unnecessary control flow, helper functions, assertions, and Python-level overhead inside the kernel and the pallas_call setup. Minimize compilation complexity.",
                ]
            )
        return opts

    def _get_pallas_isa_documentation(self, prob: Prob = None) -> str:
        return get_pallas_isa_documentation()

    def _get_prompt_rules(self, planning: bool, coding: bool, prob: Prob = None) -> str:
        is_matmul = (prob is not None
                     and getattr(prob, 'prob_type', None) == 'tpu'
                     and getattr(prob, 'prob_id', None) == 0)

        rules = [
            "The rewritten program must remain semantically equivalent to the original program, within a small numerical tolerance.",
            "Maintain correct tensor shapes, dtypes, indexing patterns, and output layout.",
            "The following imports have already been run: import jax; import jax.numpy as jnp; from jax.experimental import pallas as pl; from jax.experimental.pallas import tpu as pltpu; import numpy as np;",
            "Use pl.pallas_call() to wrap kernel functions and make them callable from JAX.",
            "Kernel functions should receive memory references (x_ref, y_ref, o_ref, etc.) and operate on them.",
        ]
        if is_matmul:
            rules = [
                "The rewritten program must remain semantically equivalent to jnp.matmul(x, y) for the harness inputs (same tolerance as the benchmark).",
                "Maintain correct tensor shapes, dtypes, and output layout matching jnp.matmul.",
                "Imports available: jax, jnp, pl, pltpu, numpy as np; you may also use: from jax import lax (e.g. for manual tiling) or jax.jit.",
                "PEAK PERFORMANCE: For dense GEMM, XLA's lowering of jnp.matmul to the MXU is often faster than hand-written Pallas. Valid solutions include: (a) test(x,y) returning jax.jit(jnp.matmul)(x,y) or jnp.matmul(x,y), (b) Pallas kernels with PrefetchScalarGridSpec / pipelined K-reduction, or (c) lax.fori_loop + dynamic_slice tiling (often slower than (a)). Prefer (a) when the goal is to match XLA reference latency.",
                "If using jax.jit, compile once (e.g. assign jitted fn to a module-level or closed-over variable) so timed runs do not retrace every call.",
                "If using Pallas, use pl.pallas_call with kernel refs as usual; bf16 compute + fp32 accumulation (jax.lax.dot with preferred_element_type=jnp.float32) is a strong pattern.",
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
        def _extract_util_percent(text: str | None) -> float | None:
            if not text:
                return None
            for line in text.split("\n"):
                l = line.strip()
                if not l:
                    continue
                m = re.search(r"(?i)\\b(utilization|util)\\b\\s*[:=]\\s*([0-9]+(?:\\.[0-9]+)?)\\s*(%)?", l)
                if not m:
                    continue
                val = float(m.group(2))
                has_pct = m.group(3) is not None
                if not has_pct and 0.0 <= val <= 1.0:
                    val *= 100.0
                if val < 0 or val > 1000:
                    continue
                return round(val, 3)
            return None

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
        include_util_feedback = random.random() < give_util_feedback

        parents_prompt = ""
        cur_cand = candidate
        while cur_cand is not None:
            # Go up to each parent and append to front of prompt
            if include_score_feedback and (cur_cand.score is not None):
                parents_prompt = f"The latency of this code was {cur_cand.score} ms.\n" + parents_prompt

            # TPU utilization feedback (if present in stdout/stderr).
            if include_util_feedback:
                util = _extract_util_percent(getattr(cur_cand, "stdout", None)) or _extract_util_percent(getattr(cur_cand, "stderr", None))
                if util is not None:
                    parents_prompt = f"The TPU utilization of this code was {util}%.\n" + parents_prompt

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
        if prob is not None and getattr(prob, 'prob_type', None) == 'tpu' and getattr(prob, 'prob_id', None) == 0:
            prompt_text += (
                "The goal is to match or beat XLA's optimized jnp.matmul on TPU (reference latency depends on M,K,N and dtype). "
                "IMPORTANT: That reference IS XLA matmul—so the fastest valid implementation is often simply jax.jit(jnp.matmul)(x,y) inside test(x,y), not a custom Pallas kernel. "
                "Hand-tuned Pallas can win for fused or irregular ops, but plain dense matmul rarely beats a single jitted jnp.matmul. "
                "If the current code is already Pallas-based and slow, consider planning a switch to XLA matmul + jit. "
            )
            prompt_text += "Do NOT repeat optimizations already applied; choose a different approach when the current path is stuck. "
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
        prompt_text += "\nYou are an expert JAX-on-TPU performance engineer. "
        if getattr(prob, "prob_type", None) == "tpu" and getattr(prob, "prob_id", None) == 0:
            prompt_text += "For matmul, jax.jit(jnp.matmul) is a first-class target, not only Pallas. "
        prompt_text += "Let's optimize the original code based on the following plan:\n"
        prompt_text += candidate.plan

        prompt_text += "\nMake sure to follow these rules:\n"
        prompt_text += self._get_prompt_rules(planning=False, coding=True, prob=prob)
        prompt_text += "\nOptimized code:"

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

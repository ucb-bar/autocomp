"""
Saturn LLM Agent for RISC-V Vector code optimization on Saturn.

This agent generates prompts for optimizing RVV code targeting the Saturn vector unit.
Supports automatic ISA documentation selection based on code analysis.
"""

import pathlib
import random
from typing import Optional

from autocomp.common import logger
from autocomp.search.prob import Prob
from autocomp.search.code_repo import CodeCandidate
from autocomp.agents.llm_agent import LLMAgent
from autocomp.agents.saturn.saturn_config import SaturnConfig
from autocomp.agents.saturn.saturn_isa_generator import SaturnIsaGenerator


class SaturnLLMAgent(LLMAgent):
    """LLM Agent for optimizing RISC-V Vector code on Saturn.
    
    Supports automatic selection of relevant ISA documentation sections
    using LLM-based code analysis. The ISA documentation is generated once
    at initialization (or on first use) and cached for all subsequent prompts.
    """

    def __init__(self, model, config: SaturnConfig = None, use_llm_isa_selection: bool = False):
        """Initialize the Saturn LLM Agent.
        
        Args:
            model: Model identifier string (e.g., "gpt-4", "claude-3-opus")
            config: Saturn hardware configuration. Uses defaults if not provided.
            use_llm_isa_selection: If True, use LLM to automatically select
                                   relevant ISA documentation sections based
                                   on the initial code being optimized.
                                   Default is False (include all sections).
        """
        super().__init__(model)
        self.config = config or SaturnConfig()
        self.use_llm_isa_selection = use_llm_isa_selection
        
        # Pass config and LLM client to ISA generator
        if use_llm_isa_selection:
            self.isa_generator = SaturnIsaGenerator(config=self.config, llm_client=self.llm_client)
        else:
            self.isa_generator = SaturnIsaGenerator(config=self.config)
        
        # Cached ISA documentation string (generated once, reused for all prompts)
        # By default, generate all sections immediately (no LLM call needed)
        if not use_llm_isa_selection:
            self._cached_isa_docs: Optional[str] = self.isa_generator.generate_isa()
        else:
            self._cached_isa_docs = None
        self._isa_selection_done: bool = not use_llm_isa_selection

    def initialize_isa_docs(self, code: str, prob: Prob = None) -> str:
        """Initialize and cache the ISA documentation based on the given code.
        
        This should be called once at the beginning of optimization with the
        initial/root code. The selected ISA sections will be cached and reused
        for all subsequent prompts.
        
        Args:
            code: The initial code to analyze for ISA section selection
            prob: Problem specification (optional)
            
        Returns:
            The generated ISA documentation string
        """
        if self._isa_selection_done and self._cached_isa_docs is not None:
            logger.debug("ISA docs already initialized, returning cached version")
            return self._cached_isa_docs
        
        logger.info("Initializing ISA documentation (one-time LLM selection)")
        self._cached_isa_docs = self.isa_generator.generate_isa(
            prob=prob,
            code=code,
            use_llm_selection=self.use_llm_isa_selection
        )
        self._isa_selection_done = True
        
        return self._cached_isa_docs

    def get_isa_docs(self, code: str = None, prob: Prob = None) -> str:
        """Get ISA documentation, using cached version if available.
        
        By default, returns all ISA documentation sections (cached at init).
        If LLM selection is enabled and docs haven't been initialized yet,
        this will trigger LLM-based section selection.
        
        Args:
            code: Code to analyze (only used for LLM selection if not yet initialized)
            prob: Problem specification (optional)
            
        Returns:
            The ISA documentation string
        """
        # If already cached, return it
        if self._cached_isa_docs is not None:
            return self._cached_isa_docs
        
        # If we have code and LLM selection is enabled, initialize with selection
        if code and self.use_llm_isa_selection:
            return self.initialize_isa_docs(code, prob)
        
        # Fallback: generate all sections (shouldn't normally reach here)
        self._cached_isa_docs = self.isa_generator.generate_isa()
        return self._cached_isa_docs

    def reset_isa_cache(self):
        """Reset the cached ISA documentation.
        
        Call this if you want to re-select ISA sections for a new optimization run.
        If LLM selection is disabled, this will regenerate all sections.
        """
        if self.use_llm_isa_selection:
            self._cached_isa_docs = None
            self._isa_selection_done = False
        else:
            # Without LLM selection, just regenerate all sections
            self._cached_isa_docs = self.isa_generator.generate_isa()
            self._isa_selection_done = True
        logger.info("ISA documentation cache reset")

    def __repr__(self):
        return f"SaturnLLMAgent({self.llm_client.model}, vlen={self.config.vlen})"

    def _get_convert_to_rvv_menu_options(self) -> list[str]:
        return [
            "convert scalar code into vectorized RVV code",
            "convert SIMD intrinsics (e.g., AVX, NEON) to RVV intrinsics",
            "vectorize a loop using RVV stripmine pattern",
            "convert explicit loop unrolling to LMUL-based vector grouping",
        ]

    def get_opt_menu_options(self, prob: Prob) -> list[str]:
        """Get optimization menu options for RVV/Saturn kernels."""
        return [
            # LMUL and register utilization
            "adjust LMUL",
            "register blocking",
            
            # Loop transformations
            "modify loop tiling",
            "loop reordering",
            "split loops",
            "fuse loops",
            "loop unrolling",
            "hoisting",
            "minimize loop overhead",
            
            # Memory optimizations
            "use unit-stride memory access",
            "use segmented loads/stores for array-of-structs",
            "prefetching",
            "double buffering",
            "minimize data movement",
            
            # Scheduling
            "interleave loads and arithmetic for chaining",
            "reorder operations for better scheduling",
            "software pipelining",
            
            # Arithmetic optimizations
            "use fused operations (e.g., vfmacc)",
            "use widening operations",
            "use scalar broadcast for outer products",
            "simplify arithmetic",
            "strength reduction (e.g., replace division with multiply)",
            
            # Reduction optimizations
            "accumulate in vector registers, reduce at end",
            "use multiple accumulators",
            
            # Permutation optimizations
            "use vslidedown and/or vslideup for regular access patterns",
            
            # Masking optimizations
            "use predicated operations",
            
            # General
            "substitute operations with equivalent operations that are faster",
            "eliminate redundant computation",
            "other methods not listed here",
        ]

    def _get_prompt_rules(self, planning: bool, coding: bool, prob: Prob = None) -> str:
        """Get rules for the prompt based on phase (planning vs coding)."""
        rules = [
            "The rewritten program should be semantically equivalent to the original program.",
            "Use proper RVV intrinsic syntax (__riscv_* functions).",
            "Maintain correct vector length handling with vsetvl stripmining pattern.",
            "Ensure proper element width and LMUL settings match data types.",
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
            rules.append("Wrap the generated code with ```c at the beginning and ``` at the end.")
            rules.append("Include necessary headers: #include <riscv_vector.h>")
        
        rules.append("Ensure vector operations respect the current vl (vector length).")
        
        # Problem-specific rules can be added here based on prob
        if prob is not None:
            # Add problem-specific shape hints if available
            pass
        
        prompt_text = ""
        for i, rule in enumerate(rules):
            prompt_text += f"{i+1}. {rule}\n"
        return prompt_text

    def _get_propose_optimizations_prompt(
        self,
        candidate: CodeCandidate,
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
        """Generate prompt for proposing optimizations."""
        
        # Select which menu options will appear
        menu_options_text = ""
        if translate:
            opt_lst = self._get_convert_to_rvv_menu_options()
        else:
            opt_lst = self.get_opt_menu_options(prob)
            if dropout_menu_options < 1 and not force_opt_menu:
                opt_lst = [opt for opt in opt_lst if random.random() < dropout_menu_options]
            if shuffle_opts:
                random.shuffle(opt_lst)
        
        include_score_feedback = random.random() < give_score_feedback

        # Build prompt from ancestors
        parents_prompt = ""
        cur_cand = candidate
        while cur_cand is not None:
            if include_score_feedback and (cur_cand.score is not None):
                parents_prompt = f"The latency of this code was {cur_cand.score} cycles.\n" + parents_prompt
            if not include_ancestors:
                parents_prompt = "\nThe original unoptimized code was:\n```c\n" + cur_cand.code + "\n```\n" + parents_prompt
                break
            elif cur_cand.plan is not None:
                parents_prompt = "\nNext, we applied this plan to the code:\n" + cur_cand.plan + "\nThe generated code was:\n```c\n" + cur_cand.code + "\n```\n" + parents_prompt
            else:
                parents_prompt = "\nThe original unoptimized code was:\n```c\n" + cur_cand.code + "\n```\n" + parents_prompt
            cur_cand = cur_cand.parent

        if analysis:
            parents_prompt += "\n" + analysis

        # Initialize prompt with RVV context (uses cached ISA docs)
        prompt_text = "RISC-V Vector (RVV) is a scalable vector extension for writing high-performance kernels on RISC-V processors.\n"
        prompt_text += "We are targeting the Saturn Vector Unit, a short-vector SIMD-style implementation optimized for DSP workloads.\n\n"
        prompt_text += self.get_isa_docs(code=candidate.code, prob=prob)
        
        prompt_text += parents_prompt

        # Add optimization menu
        for i, opt in enumerate(opt_lst):
            menu_options_text += f"{i+1}. {opt}\n"
        
        prompt_text += "\nPlease carefully review the RVV code to identify any inefficiencies. "
        prompt_text += "Performance can be improved by using the following optimizations:\n"
        prompt_text += "<optimizations>:\n" + menu_options_text + "\n"
        
        if force_opt_menu:
            prompt_text += f"Explain how to apply <optimization> {force_opt_menu}: '{opt_lst[force_opt_menu-1]}' to the above code to reduce execution time, and explain how it will improve performance."
        else:
            prompt_text += "You are an expert RVV performance engineer generating high-performance vector kernels for Saturn. "
            
            choose_or_invent = random.random()
            if choose_or_invent < 0.1 and not translate:
                prompt_text += "Invent a new optimization inspired by the <optimizations> to apply to the above code to reduce execution time, and explain how it will improve performance."
            elif choose_or_invent < 0.2 and not translate:
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

    def _get_implement_code_prompt(
        self, candidate: CodeCandidate, prob: Prob = None, code_icl_examples: bool = True
    ) -> str:
        """Generate prompt for implementing optimized code."""
        
        prompt_text = "RISC-V Vector (RVV) is a scalable vector extension for writing high-performance kernels on RISC-V processors.\n"
        prompt_text += "We are targeting the Saturn Vector Unit, a short-vector SIMD-style implementation optimized for DSP workloads.\n\n"
        
        if prob is None:
            raise ValueError("RvvLLMAgent requires prob parameter to be provided")
        
        # Use cached ISA docs
        prompt_text += self.get_isa_docs(code=candidate.parent.code, prob=prob)

        prompt_text += "\nThe original code is as follows:\n```c\n"
        prompt_text += candidate.parent.code
        prompt_text += "\n```\n"
        prompt_text += "\nYou are an expert RVV performance engineer generating high-performance vector kernels for Saturn. "
        prompt_text += "Let's optimize the original code based on the following plan:\n"
        prompt_text += candidate.plan

        prompt_text += "\nMake sure to follow these rules:\n"
        prompt_text += self._get_prompt_rules(planning=False, coding=True, prob=prob)
        prompt_text += "\nOptimized RVV code:"

        return prompt_text

    def _get_combine_candidates_prompt(
        self, candidates: list[CodeCandidate], prob: Prob = None
    ) -> str:
        """Generate prompt for combining multiple candidate implementations."""
        
        prompt_text = "RISC-V Vector (RVV) is a scalable vector extension for writing high-performance kernels on RISC-V processors.\n"
        prompt_text += "You are an expert RVV performance engineer generating high-performance vector kernels for Saturn. "
        prompt_text += "Let's combine the following optimized RVV code samples to extract the high-performance characteristics of each:\n"
        
        # Use cached ISA docs
        code_for_selection = candidates[0].code if candidates else None
        prompt_text += "\n" + self.get_isa_docs(code=code_for_selection, prob=prob) + "\n"
        
        for i, c in enumerate(candidates):
            prompt_text += f"Sample {i+1}:\n```c\n{c.code}\n```\n"

        prompt_text += "\nMake sure to follow these rules:"
        prompt_text += self._get_prompt_rules(planning=False, coding=True, prob=prob)
        prompt_text += "\nOptimized combined RVV code:"
        
        return prompt_text

    def _get_reimplement_failed_code_prompt(
        self, candidate: CodeCandidate, prob: Prob = None
    ) -> str:
        """Generate prompt to fix failed code based on error feedback."""
        
        if prob is None:
            raise ValueError("RvvLLMAgent requires prob parameter to be provided")

        prompt_text = "RISC-V Vector (RVV) is a scalable vector extension for writing high-performance kernels on RISC-V processors.\n"
        # Use cached ISA docs
        prompt_text += self.get_isa_docs(code=candidate.code, prob=prob)

        prompt_text += "\n\nYou are an expert RVV performance engineer generating high-performance vector kernels for Saturn. "
        prompt_text += "\nThe code was:\n```c\n"
        prompt_text += candidate.code
        prompt_text += "\n```\n"
        
        # Add error information
        prompt_text += "\n\nHowever, the code failed with the following output:\n"
        if candidate.stderr:
            prompt_text += "=== STDERR ===\n"
            stderr_lines = candidate.stderr.split("\n")
            stderr_lines = [line[:400] for line in stderr_lines]
            prompt_text += "\n".join(stderr_lines) + "\n"
        if candidate.stdout:
            prompt_text += "=== STDOUT ===\n"
            stdout_lines = candidate.stdout.split("\n")
            stdout_lines = [line[:400] for line in stdout_lines]
            prompt_text += "\n".join(stdout_lines) + "\n"
        
        prompt_text += "\nPlease fix the code to address the errors while still applying the optimization plan. "
        prompt_text += "Make sure to follow these rules:\n"
        prompt_text += self._get_prompt_rules(planning=False, coding=True, prob=prob)
        prompt_text += "\nFixed and optimized RVV code:"

        return prompt_text

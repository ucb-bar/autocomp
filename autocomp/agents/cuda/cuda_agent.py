import pathlib
import random

from autocomp.common import logger
from autocomp.search.prob import Prob
from autocomp.search.code_repo import CodeCandidate
from autocomp.agents.llm_agent import LLMAgent
from autocomp.agents.cuda.prompts import tensor_examples
from autocomp.hw_config.cuda_config import CudaHardwareConfig
from autocomp.backend.eval_backend import EvalBackend

class CudaLLMAgent(LLMAgent):
    def __init__(self, model, hw_config: CudaHardwareConfig, eval_backend: EvalBackend):
        super().__init__(model)
        self.hw_config = hw_config
        self.eval_backend = eval_backend

    def _get_convert_to_cuda_menu_options(self) -> list[str]:
        return [
            # "Convert PyTorch code to functional PyTorch code",
            "Convert a PyTorch operation to inline CUDA C++ code",
            # "Use CUDA Graph Capture to eliminate launch overhead",
            # "Convert a PyTorch operation to Triton code",
        ]

    def get_opt_menu_options(self, prob: Prob = None) -> list[str]:
        return [
            # "Convert PyTorch code to functional PyTorch code",
            "Convert an operation to optimized CUDA C++ code",
            "Convert an operation to CUDA C++ code",
            "Convert an operation to optimized Triton code",
            "Reduce PyTorch launch overhead",
            "Use compilation flags like -O3 and --use_fast_math when compiling CUDA code",
            # General Kernel and Memory Optimizations
            "Minimize global memory accesses",
            "Use shared memory to reduce global memory bandwidth usage",
            "Cache redundantly computed data in shared memory",
            "Use pointers to global memory rather than copying to shared memory",
            "Coalesce global memory accesses",
            "Avoid bank conflicts in shared memory",
            "Use registers efficiently; avoid register spilling",
            "Minimize divergent branches within warps",
            "Use CUDA warp-level primitives for synchronization",
            "Fuse kernels when possible to reduce kernel launch overhead",
            "Minimize number of synchronization points",
            "Store more data and reduce at the end rather than using atomic operations",
            "Use grid-stride loops",
            "Tile operations for optimal cache utilization",
            "Use L2 persisting cache window to keep frequently reused tensors resident in L2",
            "Use multiple CUDA streams to overlap computation and data movement",
            # CUDA graph-related Optimizations
            "overlap host-to-device transfers with the CUDA-Graph replay",
            # Thread and Block-Level Optimizations
            "Maximize occupancy without excessive register usage",
            "Choose optimal block sizes (typically multiples of 32 threads)",
            "Use __restrict__ to help compiler with pointer aliasing",
            "Loop unrolling (#pragma unroll)",
            # # Tensor and GEMM Specific Optimizations
            "Use cuBLASLt for Tensor Core GEMM operations",
            "Use cuBLASLt, cuBLAS, or cuDNN for GEMM and convolution operations instead of custom kernels",
            "Use Tensor Cores (e.g. wmma APIs) for mixed precision acceleration (FP16, TF32, INT8)",
            "Use PyTorch's tensor core APIs (torch.eval_backends.cuda.matmul.allow_tf32, torch.eval_backends.cudnn.allow_tf32, torch.amp) to enable Tensor Cores",
            "Use lower precision (e.g. bfloat16, float16, float8_e4m3fn) for computations",
            "Quantize weights or activations where accuracy permits (e.g. bfloat16)",
            "Leverage fused operations in cuDNN (e.g. convolution + bias + ReLU)",
            # Memory Transfer Optimizations
            "Overlap computation and data transfer using CUDA streams and asynchronous copies",
            "Use pinned (page-locked) host memory for faster host-device transfers",
            "Minimize host-device transfer frequency",
            # # Algorithmic Optimizations
            "Choose optimal convolution algorithms (FFT, Winograd, implicit GEMM) based on kernel size",
            "Prune unneeded weights for sparse computation",
            "Batch inputs to maximize GPU utilization",
            "Reuse intermediate results where possible (e.g. shared activations)",
            "Vectorize operations by using wider data types",
            "Use Tensor core GEMMs for GEMM-like operations",
            "Convert convolution operations to Tensor core GEMMs",
            # "Convert to a lower precision",
            # From CUDA-L1
            "Skip computation when data-dependent execution encounters zero values or a branch that will never be taken",
            "Ensure data is stored in contiguous memory blocks",
            "Arrange data access patterns to maximize memory bandwidth and minimize latency through techniques like shared memory usage, coalesced global memory access, and memory padding",
            "Memory Coalescing: optimize CUDA kernel performance by ensuring threads in the same warp access contiguous memory locations",
            "Pre-allocate input and output tensors during graph initialization and reuse them",
            "Merge low-level operations",
            "Merge high-level operations",
            "Reorder operations or blocks of operations",
            "Hoist redundant operations out of loops",
            "Substitute operations with equivalent operations that are faster",
            "Double buffering",
            "Pipeline operations to better overlap computation and data movement",
            "Minimize data movement",
            # Other random stuff
            "Use built-in CUDA primitive functions",
            "Call torch:: functions from C++ rather than from Python",
            "Use ATen at:: functions rather than PyTorch functions",
            "Use CUDA graph capture",
            "Use dedicated CUDA streams",
            "Profile the code and capture CUDA graphs in the __init__ function",
            "Simplify operations where possible",
            "Classical compiler optimizations",
            "Any other optimizations that you think are relevant",
        ]

    def analyze_code(self, candidate: CodeCandidate, num_to_gen: int, save_dir: pathlib.Path, save_str: str) -> list[str]:
        return []
    
    def _get_prompt_rules(self, planning: bool, coding: bool) -> str:
        rules = []
        rules.extend(self.hw_config.get_hw_config_specific_rules())
        rules.extend(self.eval_backend.get_backend_specific_rules())
        rules.extend([
            "The rewritten program should be semantically equivalent to the original program, within a small numerical tolerance.",
            "Do not add fallback paths that revert to the original code.",
        ])
        if planning:
            rules.append("Limit the scope of the plan to the selected optimization.")
        if coding:
            rules.append("Wrap the generated code with ```python at the beginning and ``` at the end.")
        rules_text = ""
        for i, rule in enumerate(rules):
            rules_text += f"{i+1}. {rule}\n"
        return rules_text

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
        if translate:
            opt_lst = self._get_convert_to_cuda_menu_options()
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
        # prompt_text = tensor_examples.PROMPT()
        prompt_text = parents_prompt

        # Now add the actual planning prompt
        menu_options_text = ""
        for i, opt in enumerate(opt_lst):
            menu_options_text += f"{i+1}. {opt}\n"
        prompt_text += """Please carefully review the program to identify any inefficiencies. 
Speedup can be increased by using the following optimizations:
<optimizations>: \n""" + menu_options_text + "\n"
        
        prompt_text += "You are an expert GPU performance engineer generating high-performance PyTorch and CUDA code. "
        if force_opt_menu:
            prompt_text += "Explain how to apply <optimization> " + str(force_opt_menu) + ": '" + opt_lst[force_opt_menu-1] + "' to the above code to reduce execution time, and explain how it will improve performance."
        else:
            if random.random() < 0.1:
                prompt_text += "Invent an optimization different from the <optimizations> above to address the inefficiencies of the above code and reduce its execution time, and explain how it will improve performance."
            else:
                prompt_text += "Come up with a plan to apply exactly one of the <optimizations> to address the inefficiencies of the above code and reduce its execution time. The plan should be specific to this code and explain how to change it."

        prompt_text += "\nMake sure to follow these rules:"
        prompt_text += self._get_prompt_rules(planning=True, coding=False)

        if prompt_end:
            logger.debug("Appended the following as prompt_end: '%s'", prompt_end)
            prompt_text += "\n" + prompt_end
        return prompt_text


    def _get_implement_code_prompt(self, candidate: CodeCandidate, prob: Prob = None, code_icl_examples: bool = True) -> str:
        prompt_text = ""
        if "tensor core" in candidate.plan.lower():
            if random.random() < 0.5:
                prompt_text += tensor_examples.PROMPT() + "\n"
        prompt_text += "\nThe original code is as follows:\n```python\n"
        prompt_text += candidate.parent.code
        prompt_text += "\n```\nYou are an expert GPU performance engineer generating high-performance PyTorch and CUDA code. Let's optimize the original code based on the following plan:\n"
        prompt_text += candidate.plan

        # # # TODO: For certain optimizations, add more context helping it implement prompt correctly.
        # # # e.g. for tiling, add examples of how to tile the code.
        # if code_icl_examples:
        #     if "tiling" in candidate.plan:
        #         prompt_text += "\n" + tiling_example.PROMPT()
        #     if " gate" in candidate.plan or "Gate" in candidate.plan:
        #         # prompt_text += "\n" + if_example_matmul.PROMPT()
        #         prompt_text += "\n" + if_example.PROMPT()

        prompt_text += "\nMake sure to follow these rules:"
        prompt_text += self._get_prompt_rules(planning=False, coding=True)

        prompt_text += "Optimized code:"

        return prompt_text

    def _get_combine_candidates_prompt(self, candidates: list[CodeCandidate], prob: Prob = None) -> str:
        prompt_text = "You are an expert GPU performance engineer generating high-performance PyTorch and CUDA code. Let's combine the following optimized code samples to extract the high-performance characteristics of each:\n"
        for i, c in enumerate(candidates):
            prompt_text += f"Sample {i+1}:\n{c.code}\n"

        prompt_text += "\nMake sure to follow these rules:"
        prompt_text += self._get_prompt_rules(planning=False, coding=True)
        prompt_text += "Optimized code:"
        return prompt_text

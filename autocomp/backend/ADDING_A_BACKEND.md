# Adding a New Backend to Autocomp

### ⚠️ We plan to make the process of adding a new backend easier and more streamlined in the future. This guide is a work in progress.

This guide explains how to add support for a new hardware backend to Autocomp. Adding a backend involves implementing several components that integrate with Autocomp's search and optimization infrastructure.

## Overview

To add a new backend, you need to:

1. [**Create a Hardware Backend class**](ADDING_A_BACKEND.md#step-1-create-a-hardware-backend-class) - Implements code evaluation and testing
2. [**Create an LLM Agent class**](ADDING_A_BACKEND.md#step-2-create-an-llm-agent-class) - Handles backend-specific prompting for code generation
3. [**Register the backend**](ADDING_A_BACKEND.md#step-3-register-the-backend) - Add backend instantiation logic in `search.py`
4. [**Create setup documentation**](ADDING_A_BACKEND.md#step-4-create-setup-documentation) - Setup instructions for users
5. [**Update README**](ADDING_A_BACKEND.md#step-5-update-readme) - Add your backend to the README

## Step 1: Create a Hardware Backend Class

Create a new directory `autocomp/backend/{backend_name}/` with a file `{backend_name}_eval.py` that implements a class inheriting from `HardwareBackend`.

### Required Implementation

Your backend class must implement the `evaluate_code` method:

```python
from typing import List
from autocomp.backend.hardware_backend import HardwareBackend
from autocomp.search.prob import Prob

class YourHardwareBackend(HardwareBackend):
    def evaluate_code(self, prob: Prob, code_strs: list[str], simulator: str) -> List[dict]:
        """
        Evaluate code candidates and return results.
        
        Args:
            prob: Problem instance containing metadata (prob_type, prob_id, etc.)
            code_strs: List of code strings to evaluate
            simulator: Simulator/evaluation method to use (e.g., "firesim", "spike", "trn")
        
        Returns:
            List of dictionaries, one per code_str, each containing:
            - "correct": bool - Whether the code passed correctness tests
            - Performance metrics (for all existing backends, this is "latency"), but new metrics can be added
        """
        results = []
        for code_str in code_strs:
            # 1. Prepare code for evaluation (may need to wrap in test harness)
            # 2. Run code through simulator/evaluator
            # 3. Parse results and extract metrics
            # 4. Run correctness tests
            result = {
                "correct": True,  # or False
                # Add your metrics here, e.g.:
                "latency": 123.45,
            }
            results.append(result)
        return results
```

### Code Cleaning

Backends generally need to clean LLM-generated code. There is some simple extraction logic implemented at the top of `autocomp/agents/llm_agent.py`, but if this does not work for your backend, you can implement your own cleaning logic in the `evaluate_code` method of your hardware backend class or in `llm_agent.py`. You can also prompt the LLM to generate only the code and no other text, but this may actually reduce the quality of the code generated!

### Examples

- **Gemmini**: See `autocomp/backend/gemmini/gemmini_eval.py` - Uses FireSim or Spike for evaluation
- **Trainium**: See `autocomp/backend/trn/trn_eval.py` - Uses Trainium Neuron runtime
- **CUDA/KernelBench**: See `autocomp/backend/kernelbench/kb_eval.py` - Uses KernelBench for evaluation

### Other Considerations

- **Test Integration**: Tests are located in `tests/{prob_type}/` and should be integrated into your evaluation
- **Error Handling**: You can consider reading in `stdout` and `stderr` from the code execution to provide feedback to the LLM, to try to fix errors automatically (like we do with Trainium), or for manual inspection later.
- **Evaluation Time**: When the evaluation is slow, this can be a bottleneck in search. Consider parallelizing or batching evaluations if it can help reduce evaluation time.

## Step 2: Create an LLM Agent Class

Create a new directory `autocomp/agents/{backend_name}/` with a file `{backend_name}_agent.py` that contains a class inheriting from `LLMAgent` (defined in `autocomp/agents/llm_agent.py`).

### Required Implementation

Your agent class must override key methods for backend-specific prompting:

```python
# In autocomp/agents/{backend_name}/{backend_name}_agent.py
from autocomp.agents.llm_agent import LLMAgent
from autocomp.search.prob import Prob
from autocomp.search.code_repo import CodeCandidate

class YourLLMAgent(LLMAgent):
    def __init__(self, model):
        super().__init__(model)
        # Initialize backend-specific configuration
    
    def get_opt_menu_options(self, prob: Prob):
        """
        Return list of optimization options available for this backend.
        These appear in the planning prompt menu.
        """
        return [
            "Option 1: Description",
            "Option 2: Description",
            # ... more options
        ]
    
    def _get_propose_optimizations_prompt(self, candidate: CodeCandidate, prob: Prob, 
                                         force_opt_menu, prompt_end, analysis, 
                                         shuffle_opts, give_score_feedback, 
                                         give_util_feedback, give_spad_acc_feedback, 
                                         include_ancestors, plan_icl_examples, 
                                         cur_iter, num_iters, dropout_menu_options,
                                         translate=False):
        """
        Generate the planning phase prompt. This includes:
        - The backend's ISA/API and architectural information
        - The current code (candidate.code)
        - Available optimizations (from get_opt_menu_options())
        - Code structure and constraints
        """
        # Build your prompt here
        prompt = f"""
        Your backend-specific planning prompt...
        """
        return prompt
    
    def _get_implement_code_prompt(self, candidate: CodeCandidate, prob: Prob, 
                                   code_icl_examples: bool = True):
        """
        Generate the implementation phase prompt. This includes:
        - The backend's ISA/API and architectural information
        - The optimization plan to implement (candidate.plan)
        - The current code (candidate.parent.code)
        - Backend-specific code format requirements
        - Examples of correct code (from autocomp/agents/{backend_name}/prompts/)
        """
        # Build your prompt here
        prompt = f"""
        Your backend-specific implementation prompt...
        """
        return prompt
```

### Examples

- **Gemmini**: `autocomp/agents/gemmini/gemmini_agent.py` - `GemminiLLMAgent`
- **Trainium**: `autocomp/agents/trn/trn_agent.py` - `TrnLLMAgent`
- **CUDA**: `autocomp/agents/cuda/cuda_agent.py` - `CudaLLMAgent`

### Other Considerations

- **Prompt Files**: Store backend-specific prompts and examples in `autocomp/agents/{backend_name}/prompts/`. This can include ISA documentation, code examples, and rules.
- **Conditional Execution and Prompt Generation**: The prompts generated can be conditional on the plans generated (for the implementation phase) or on things like random.random() (for techniques like optimization menu dropout).
- **ISA Documentation**: Include important ISA/API documentation and architectural information in prompts. The specific amount and info needed will depend on the backend. For Gemmini, we provide function signatures and descriptions for all functions in the ISA, located in `autocomp/agents/gemmini/prompts/`. For Trainium, we provide a subset of NKI instructions, using the NKI ISA generator from `autocomp/agents/trn/nki_isa_generator.py`. For CUDA, we provide tensor examples from `autocomp/agents/cuda/prompts/`, but no ISA documentation.
- **Optimization Menu**: Define backend-specific optimizations (tiling, fusion, etc.). Remember to implement dropout inside `_get_propose_optimizations_prompt()`.
- **Examples**: You may want to include in-context learning examples of optimized code.
- **Rules**: Define constraints and correctness requirements. Specify exact code format expected (function signatures, wrappers, etc.).

## Step 3: Register the Backend

Add backend and LLM agent instantiation logic in `autocomp/search/search.py` in the `main()` function.

### Import Your Classes

Add imports at the top of `search.py`:

```python
from autocomp.backend.{backend_name}.{backend_name}_eval import YourHardwareBackend
from autocomp.agents.{backend_name}.{backend_name}_agent import YourLLMAgent
```

### Add Backend Instantiation

In the `main()` function, add an `elif` clause for your backend:

```python
elif backend == "your_backend_name":
    hw_backend = YourHardwareBackend()  # Add any required parameters
    llm = LLMEnsemble([YourLLMAgent(model) for model in models])
```

### Handle Initial Code Loading

Add logic to load initial code from `sols/` directory:

```python
elif backend == "your_backend_name":
    # Load initial code from sols directory
    sol_dir = pathlib.Path(__file__).parent.parent.parent / "sols" / prob.prob_type
    # Define pattern matching for your backend's file naming convention
    matches = list(sol_dir.glob(f"{prob_id}_*.{extension}"))
    if not matches:
        raise FileNotFoundError(f"No solution file found...")
    with open(matches[0]) as f:
        initial_code = f.read()
```

## Step 4: Create Setup Documentation

Create a setup file `autocomp/backend/{backend_name}/{backend_name}_setup.md` that explains how to set up and run the backend, following the pattern of existing setup files.

See `autocomp/backend/gemmini/gemmini_setup.md`, `autocomp/backend/trn/trn_setup.md`, or `autocomp/backend/kernelbench/kb_setup.md` for examples.

## Step 5: Update README

Add your backend to the README.md:

1. **Backend Setup section**: Add link to your setup file
2. **Usage section**: Document your backend name, simulator options, and problem types
3. **Repository Structure**: Document your backend files

## Testing Your Backend

1. **Create test cases**: Add test files in `tests/{prob_type}/` matching your problem types
2. **Create baseline solutions**: Add baseline code in `sols/{prob_type}/`
3. **Run a simple optimization**: Test with a small problem to verify end-to-end functionality
4. **Check evaluation**: Verify that `evaluate_code` correctly extracts metrics and test results

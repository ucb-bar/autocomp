# Adding a New Hardware Target to Autocomp

This guide explains how to add support for a new hardware target to Autocomp. Adding a hardware target involves implementing several components that integrate with Autocomp's search and optimization infrastructure.

## Overview

To add a new hardware target, you need to:

1. [**Create a Hardware Config class**](ADDING_HARDWARE_SUPPORT.md#step-1-create-a-hardware-config-class) - Describes the target hardware
2. [**Create an Eval Backend class**](ADDING_HARDWARE_SUPPORT.md#step-2-create-an-eval-backend-class) - Implements code evaluation and testing
3. [**Create an LLM Agent class**](ADDING_HARDWARE_SUPPORT.md#step-3-create-an-llm-agent-class) - Handles target-specific prompting for code generation
4. [**Register the hardware target**](ADDING_HARDWARE_SUPPORT.md#step-4-register-the-hardware-target) - Add instantiation logic in `search.py`
5. [**Create setup documentation**](ADDING_HARDWARE_SUPPORT.md#step-5-create-setup-documentation) - Setup instructions for users
6. [**Update README**](ADDING_HARDWARE_SUPPORT.md#step-6-update-readme) - Add your hardware target to the README

## Step 1: Create a Hardware Config Class

Create a new file `autocomp/hw_config/{backend_name}_config.py` that implements a class inheriting from `HardwareConfig` (defined in `autocomp/hw_config/hardware_config.py`). This class describes the target hardware and provides hardware-specific rules for LLM prompts.

### Required Implementation

```python
from autocomp.hw_config.hardware_config import HardwareConfig

class YourHardwareConfig(HardwareConfig):
    def __init__(self, ...):
        # Store hardware-specific parameters
        pass

    def get_hw_config_specific_rules(self) -> list[str]:
        """Return a list of hardware-config-specific rule strings for LLM prompts."""
        return [
            "Your hardware-specific rule 1",
            "Your hardware-specific rule 2",
        ]

    def get_hw_description(self) -> str:
        """Return a short hardware description string for display/logging."""
        return "Your Hardware (param1, param2)"
```

Then add the import to `autocomp/hw_config/__init__.py`.

### Examples

- **CUDA**: `autocomp/hw_config/cuda_config.py` - `CudaHardwareConfig(gpu_name, pytorch_version, cuda_version)`
- **Gemmini**: `autocomp/hw_config/gemmini_config.py` - `GemminiHardwareConfig(pe_dim, spad_size_kb, acc_size_kb)`
- **Trainium**: `autocomp/hw_config/trn_config.py` - `TrnHardwareConfig(instance_type)`

## Step 2: Create an Eval Backend Class

Create a new directory `autocomp/backend/{backend_name}/` with a file `{backend_name}_eval.py` that implements a class inheriting from `EvalBackend` (defined in `autocomp/backend/eval_backend.py`).

### Required Implementation

Your eval backend class must implement the `evaluate_code` method:

```python
from typing import List
from autocomp.backend.eval_backend import EvalBackend
from autocomp.search.prob import Prob

class YourEvalBackend(EvalBackend):
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
            - Performance metrics (for all existing hardware targets, this is "latency"), but new metrics can be added
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

You can also optionally override `get_backend_specific_rules()` to return a list of backend-specific rule strings for LLM prompts, and `get_hw_feedback()` to return hardware feedback for candidates.

### Code Cleaning

Eval backends generally need to clean LLM-generated code. There is some simple extraction logic implemented at the top of `autocomp/agents/llm_agent.py`, but if this does not work for your hardware target, you can implement your own cleaning logic in the `evaluate_code` method of your eval backend class or in `llm_agent.py`. You can also prompt the LLM to generate only the code and no other text, but this may actually reduce the quality of the code generated!

### Examples

- **Gemmini**: See `autocomp/backend/gemmini/gemmini_eval.py` - Uses FireSim or Spike for evaluation
- **Trainium**: See `autocomp/backend/trn/trn_eval.py` - Uses Trainium Neuron runtime
- **CUDA/KernelBench**: See `autocomp/backend/kernelbench/kb_eval.py` - Uses KernelBench for evaluation
- **CUDA/GPU MODE**: See `autocomp/backend/gpumode/gpumode_eval.py` - Uses GPU MODE leaderboard for evaluation

### Other Considerations

- **Test Integration**: Tests are located in `tests/{prob_type}/` and should be integrated into your evaluation
- **Error Handling**: You can consider reading in `stdout` and `stderr` from the code execution to provide feedback to the LLM, to try to fix errors automatically (like we do with Trainium), or for manual inspection later.
- **Evaluation Time**: When the evaluation is slow, this can be a bottleneck in search. Consider parallelizing or batching evaluations if it can help reduce evaluation time.

## Step 3: Create an LLM Agent Class

Create a new directory `autocomp/agents/{backend_name}/` with a file `{backend_name}_agent.py` that contains a class inheriting from `LLMAgent` (defined in `autocomp/agents/llm_agent.py`).

### Required Implementation

Your agent class must override key methods for target-specific prompting:

```python
# In autocomp/agents/{backend_name}/{backend_name}_agent.py
from autocomp.agents.llm_agent import LLMAgent
from autocomp.search.prob import Prob
from autocomp.search.code_repo import CodeCandidate
from autocomp.hw_config.{backend_name}_config import YourHardwareConfig
from autocomp.backend.eval_backend import EvalBackend

class YourLLMAgent(LLMAgent):
    def __init__(self, model, hw_config: YourHardwareConfig, eval_backend: EvalBackend):
        super().__init__(model)
        self.hw_config = hw_config
        self.eval_backend = eval_backend
    
    def _get_propose_optimizations_prompt(self, candidate: CodeCandidate, prob: Prob, 
                                         force_opt_menu, prompt_end, analysis, 
                                         shuffle_opts, give_score_feedback, 
                                         give_util_feedback, give_spad_acc_feedback, 
                                         include_ancestors, plan_icl_examples, 
                                         cur_iter, num_iters, dropout_menu_options,
                                         translate=False):
        """
        Generate the planning phase prompt. This includes:
        - The hardware target's ISA/API and architectural information
        - The current code (candidate.code)
        - Available optimizations (from get_opt_menu_options())
        - Code structure and constraints
        """
        # Build your prompt here
        prompt = f"""
        Your target-specific planning prompt...
        """
        return prompt
    
    def _get_implement_code_prompt(self, candidate: CodeCandidate, prob: Prob, 
                                   code_icl_examples: bool = True):
        """
        Generate the implementation phase prompt. This includes:
        - The hardware target's ISA/API and architectural information
        - The optimization plan to implement (candidate.plan)
        - The current code (candidate.parent.code)
        - Target-specific code format requirements
        - Examples of correct code (from autocomp/agents/{backend_name}/prompts/)
        """
        # Build your prompt here
        prompt = f"""
        Your target-specific implementation prompt...
        """
        return prompt
```

### Examples

- **Gemmini**: `autocomp/agents/gemmini/gemmini_agent.py` - `GemminiLLMAgent`
- **Trainium**: `autocomp/agents/trn/trn_agent.py` - `TrnLLMAgent`
- **CUDA**: `autocomp/agents/cuda/cuda_agent.py` - `CudaLLMAgent` (used for both KernelBench and GPU MODE backends)

### Other Considerations

- **Hardware Config and Eval Backend**: The agent receives a `hw_config` object (for hardware-specific rules/descriptions) and an `eval_backend` object (for backend-specific rules). Use `self.hw_config.get_hw_config_specific_rules()` and `self.eval_backend.get_backend_specific_rules()` to include these in prompts.

- **Prompt Files**: Store target-specific prompts and examples in `autocomp/agents/{backend_name}/prompts/`. This can include ISA documentation, code examples, and rules.
- **Conditional Execution and Prompt Generation**: The prompts generated can be conditional on the plans generated (for the implementation phase) or on things like random.random() (for techniques like optimization menu dropout).
- **ISA Documentation**: Include important ISA/API documentation and architectural information in prompts. The specific amount and info needed will depend on the hardware target. For Gemmini, we provide function signatures and descriptions for all functions in the ISA, located in `autocomp/agents/gemmini/prompts/`. For Trainium, we provide a subset of NKI instructions, using the NKI ISA generator from `autocomp/agents/trn/nki_isa_generator.py`. For CUDA, we provide tensor examples from `autocomp/agents/cuda/prompts/`, but no ISA documentation.
- **Optimization Menu**: Define target-specific optimizations (tiling, fusion, etc.). Remember to implement dropout inside `_get_propose_optimizations_prompt()`.
- **Examples**: You may want to include in-context learning examples of optimized code.
- **Rules**: Define constraints and correctness requirements. Specify exact code format expected (function signatures, wrappers, etc.).

## Step 4: Register the Hardware Target

Register your hardware target's components in the helper functions in `autocomp/search/search.py`.

### Import Your Classes

Add imports at the top of `search.py`:

```python
from autocomp.backend.{backend_name}.{backend_name}_eval import YourEvalBackend
from autocomp.agents.{backend_name}.{backend_name}_agent import YourLLMAgent
from autocomp.hw_config.{backend_name}_config import YourHardwareConfig
```

### Add Eval Backend and Agent Instantiation

Add an `elif` clause for your eval backend and agent in the `create_backend_and_agents()` function:

```python
def create_backend_and_agents(backend_name: str, agent_name: str, hw_config, prob: "Prob", models: list, code_models: list = None):
    ...
    # Create eval backend
    elif backend_name == "your_backend":
        eval_backend = YourEvalBackend()  # Add any required parameters
    ...
    # Create agents
    elif agent_name == "your_agent":
        agent = LLMEnsemble([YourLLMAgent(m, hw_config, eval_backend) for m in models])
        code_agent = LLMEnsemble([YourLLMAgent(m, hw_config, eval_backend) for m in code_models]) if code_models else None
    ...
```

### Add a Hardware Config Instantiation

In the `main()` function of `search.py`, add an example of instantiating your hardware config:

```python
# hw_config = YourHardwareConfig(...)
```

### Handle Initial Code Loading

Add logic to load initial code in the `load_initial_code()` function:

```python
def load_initial_code(backend_name: str, prob: "Prob") -> str:
    ...
    elif backend_name == "your_backend":
        sol_dir = SOLS_DIR / prob_type
        matches = list(sol_dir.glob(f"{prob_id}_*.{extension}"))
        if not matches:
            raise FileNotFoundError(f"No file matching {prob_id}_*.{extension} in {sol_dir}")
        with open(matches[0]) as f:
            return f.read()
    ...
```

## Step 5: Create Setup Documentation

Create a setup file `autocomp/backend/{backend_name}/{backend_name}_setup.md` that explains how to set up and run the hardware target, following the pattern of existing setup files.

See `autocomp/backend/gemmini/gemmini_setup.md`, `autocomp/backend/trn/trn_setup.md`, or `autocomp/backend/kernelbench/kb_setup.md` for examples.

## Step 6: Update README

Add your hardware target to the README.md:

1. **Hardware Target Setup section**: Add link to your setup file
2. **Usage section**: Document your `backend_name`, simulator options, and problem types
3. **Repository Structure**: Document your files

## Testing Your Hardware Target

1. **Create test cases**: Add test files in `tests/{prob_type}/` matching your problem types
2. **Create baseline solutions**: Add baseline code in `sols/{prob_type}/`
3. **Run a simple optimization**: Test with a small problem to verify end-to-end functionality
4. **Check evaluation**: Verify that `evaluate_code` correctly extracts metrics and test results

# Contributing to Autocomp

## Development Setup

```bash
pip install -e ".[dev]"
```

This installs the package in editable mode along with test dependencies (currently just `pytest`).

## Running Tests

```bash
WANDB_MODE=disabled pytest
```

`WANDB_MODE=disabled` prevents wandb from requiring credentials. Tests are discovered from `autocomp_tests/` (configured in `pyproject.toml`).

## How Tests Work

### Dummy LLM Provider

`LLMClient` supports a `"dummy"` provider that returns canned strings without making any API calls or requiring credentials. To use it, pass `"dummy::any-model-name"` as the model string:

```python
from autocomp.agents.trn.trn_agent import TrnLLMAgent

agent = TrnLLMAgent("dummy::test-model", hw_config, eval_backend)
# agent.llm_client.chat(...) returns ["dummy response", ...]
# agent.llm_client.chat_async(...) returns [["dummy response", ...], ...]
```

This flows through all real constructors -- no monkeypatching needed.

### DummyEvalBackend

For evaluation, `autocomp_tests/integration/conftest.py` provides a `DummyEvalBackend` that always returns `{"correct": True, "p99_latency": 1.0}`. This lets the full search loop run without hardware.

### Adding a New Test

1. Add your test file under `autocomp_tests/integration/`.
2. Use `"dummy::test-model"` for agents and `DummyEvalBackend` for evaluation (both available via fixtures in `conftest.py`).
3. Verify locally: `WANDB_MODE=disabled pytest autocomp_tests/ -v`

## CI

GitHub Actions runs `pytest` on every push to `main` and on every PR. The workflow is defined in `.github/workflows/ci.yml`. It installs the package with `pip install -e ".[dev]"` and runs with `WANDB_MODE=disabled`.

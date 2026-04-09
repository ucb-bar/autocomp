"""Common functions and utilities."""
import pathlib

from .my_logging import logger
from .llm_utils import LLMClient

REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
HARNESSES_DIR = REPO_ROOT / "harnesses"
SOLS_DIR = REPO_ROOT / "sols"
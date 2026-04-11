"""
Test harness for Problem 0: 32x32 Matrix Multiplication (fp8 -> bf16)

PyTorch reference:
    C = A @ B
    where A is (32, 32) float8_e4m3fn, B is (32, 32) float8_e4m3fn,
    and C is (32, 32) bfloat16.

Memory layout:
    DRAM_ACTIVATION_BASE = 0x0000  — A (32x32 fp8, 1024 bytes)
    DRAM_WEIGHT_BASE     = 0x0400  — B (32x32 fp8, 1024 bytes)
    DRAM_OUTPUT_BASE     = 0x0800  — C (32x32 bf16, 2048 bytes)
"""
from typing import List, Tuple, Any

import torch
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs
from npu_model.software import Instruction, Program

# ── Memory layout constants (problem-owned) ──────────────────────────
DRAM_ACTIVATION_BASE = 0x0000
DRAM_WEIGHT_BASE = 0x0400
DRAM_OUTPUT_BASE = 0x0800

# ── Input data (problem-owned) ───────────────────────────────────────
ACTIVATION_DATA = torch.eye(32, 32, dtype=torch.float8_e4m3fn)
WEIGHT_DATA = (2 * torch.eye(32, 32, dtype=torch.float32)).to(torch.float8_e4m3fn)

# ── Golden output (computed from PyTorch reference) ──────────────────
_MATMUL_RESULT = (
    ACTIVATION_DATA.to(torch.float32) @ WEIGHT_DATA.to(torch.float32)
).to(torch.bfloat16)

GOLDEN_OUTPUT = torch.cat((_MATMUL_RESULT[:, :16], _MATMUL_RESULT[:, 16:]), dim=0)

# SUBSTITUTE HERE
def test() -> List[Instruction[Any]]:
    return []
# SUBSTITUTE END

class TestProgram(Program):
    """Assembled by the test harness — do not modify."""
    instructions: List[Instruction[Any]] = test()
    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_ACTIVATION_BASE, ACTIVATION_DATA),
        (DRAM_WEIGHT_BASE, WEIGHT_DATA),
    ]
    golden_result: Tuple[int, torch.Tensor] = (DRAM_OUTPUT_BASE, GOLDEN_OUTPUT)

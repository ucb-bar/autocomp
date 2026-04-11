## smolvla_silu.py

SUMMARY: Demonstrates how to define a SmolVLA NPU kernel program by mapping MLIR operations to NPU ISA instructions, using SiLU activation as a template with MLIR definition, PyTorch reference implementation, and statically-scheduled instruction sequences.

```python
from typing import Any, List, Tuple
import torch
from ...software import Instruction, Program
from npu_model.isa import DmaArgs, ScalarArgs, VectorArgs

class SmolVLASiluProgram(Program):
    """SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))."""

    instructions: List[Instruction[Any]] = [
        # ── Scalar register setup ──
        Instruction(mnemonic="addi", args=ScalarArgs(rd=1, rs1=0, imm=VMEM_INPUT_BASE)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=2, rs1=0, imm=VMEM_OUTPUT_BASE)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=3, rs1=0, imm=DRAM_INPUT_BASE)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=4, rs1=0, imm=DRAM_OUTPUT_BASE)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=5, rs1=0, imm=TILE_BYTES)),
        # ── DMA: DRAM → VMEM ──
        Instruction(mnemonic="dma.config.ch<N>", args=DmaArgs(rs1=0, channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(mnemonic="dma.load.ch<N>", args=DmaArgs(rd=1, rs1=3, rs2=5, channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        # ── Load input to MRF + constants ──
        Instruction(mnemonic="vload", args=VectorArgs(vd=0, rs1=1, imm12=0)),
        Instruction(mnemonic="vli.all", args=VectorArgs(vd=1, imm=-1)),
        Instruction(mnemonic="vli.all", args=VectorArgs(vd=2, imm=1)),
        # ── SiLU: x / (1 + exp(-x)) ──
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=3, vs1=0, vs2=1)),
        Instruction(mnemonic="vexp.bf16", args=VectorArgs(vd=4, vs1=3)),
        Instruction(mnemonic="vadd.bf16", args=VectorArgs(vd=5, vs1=4, vs2=2)),
        Instruction(mnemonic="vrecip.bf16", args=VectorArgs(vd=6, vs1=5)),
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=7, vs1=0, vs2=6)),
        # ── Store: MRF → VMEM → DRAM ──
        Instruction(mnemonic="vstore", args=VectorArgs(vd=7, rs1=2, imm12=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=20)),
        Instruction(mnemonic="dma.store.ch<N>", args=DmaArgs(rd=4, rs1=2, rs2=5, channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_INPUT_BASE, INPUT),
    ]

    golden_result: tuple[int, torch.Tensor] = (
        DRAM_OUTPUT_BASE,
        EXPECTED,
    )
```

```python
def silu_reference(x: torch.Tensor) -> torch.Tensor:
    """SiLU(x) = x * sigmoid(x). Matches the MLIR linalg.generic above."""
    return (x.float() * torch.sigmoid(x.float())).to(x.dtype)
```

## gemma_mlp.py

SUMMARY: Demonstrates how to define a statically-scheduled NPU kernel program for Gemma MLP operations by composing scalar, vector, and matrix instructions with explicit memory layout management across DRAM and VMEM, including DMA transfers, weight loading, matrix multiplications, and element-wise operations.

```python
from typing import List, Tuple
from ...software import (
    Instruction,
    Program,
)
import torch
from npu_model.isa import DmaArgs, MatrixArgs, VectorArgs, ScalarArgs
from npu_model.workload.gemma_blocks import gemma_mlp_gate_up_forward


class GemmaMlpProgram(Program):
    """
    Gemma MLP kernel program (simplified).
    Gate and up projections, then elementwise gate*up (simplified GeGLU).
    """

    instructions: List[Instruction] = [
        # Setup VMEM base addresses using LUI + ADDI for 12-bit immediates
        Instruction(mnemonic="lui", args=ScalarArgs(rd=1, imm=0x2)),
        Instruction(mnemonic="lui", args=ScalarArgs(rd=2, imm=0x2)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=2, rs1=2, imm=0x400)),
        Instruction(mnemonic="lui", args=ScalarArgs(rd=3, imm=0x3)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=3, rs1=3, imm=-2048)),
        Instruction(mnemonic="lui", args=ScalarArgs(rd=4, imm=0x3)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=4, rs1=4, imm=-1024)),
        
        # Setup DRAM base addresses
        Instruction(mnemonic="addi", args=ScalarArgs(rd=5, rs1=0, imm=0x0000)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=6, rs1=0, imm=0x0400)),
        Instruction(mnemonic="lui", args=ScalarArgs(rd=7, imm=0x1)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=7, rs1=7, imm=-2048)),
        Instruction(mnemonic="lui", args=ScalarArgs(rd=8, imm=0x1)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=8, rs1=8, imm=-1024)),
        
        # Setup transfer sizes
        Instruction(mnemonic="addi", args=ScalarArgs(rd=9, rs1=0, imm=1024)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=10, rs1=0, imm=1024)),

        # DMA: Load weights and activation from DRAM to VMEM
        Instruction(mnemonic="dma.config.ch<N>", args=DmaArgs(rs1=0, channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(mnemonic="dma.load.ch<N>", args=DmaArgs(rd=1, rs1=5, rs2=9, channel=0)),
        Instruction(mnemonic="dma.load.ch<N>", args=DmaArgs(rd=2, rs1=6, rs2=9, channel=1)),
        Instruction(mnemonic="dma.load.ch<N>", args=DmaArgs(rd=3, rs1=7, rs2=9, channel=2)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=1)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=2)),

        # Load vectors from VMEM to MRF
        Instruction(mnemonic="vload", args=VectorArgs(vd=0, rs1=1, imm12=0)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=1, rs1=2, imm12=0)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=2, rs1=3, imm12=0)),

        # Push weights to MXU0 weight buffer
        Instruction(mnemonic="vmatpush.weight.mxu0", args=VectorArgs(vd=0, vs1=0)),
        Instruction(mnemonic="vmatpush.weight.mxu0", args=VectorArgs(vd=1, vs1=1)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=17)),

        # Gate projection: activation @ gate_weight
        Instruction(mnemonic="vmatmul.mxu0", args=MatrixArgs(vd=0, vs1=2, vs2=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=33)),
        Instruction(mnemonic="vmatpop.bf16.acc.mxu0", args=VectorArgs(vd=4, vs1=0)),
        
        # Up projection: activation @ up_weight
        Instruction(mnemonic="vmatmul.mxu0", args=MatrixArgs(vd=0, vs1=2, vs2=1)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=33)),
        Instruction(mnemonic="vmatpop.bf16.acc.mxu0", args=VectorArgs(vd=6, vs1=0)),
        
        # Element-wise multiplication (GeGLU: gate * up)
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=8, vs1=4, vs2=6)),
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=9, vs1=5, vs2=7)),
        
        # Store results to VMEM
        Instruction(mnemonic="vstore", args=VectorArgs(vd=8, rs1=4, imm12=0)),
        Instruction(mnemonic="vstore", args=VectorArgs(vd=9, rs1=4, imm12=32)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=40)),

        # DMA: Store results from VMEM back to DRAM
        Instruction(mnemonic="addi", args=ScalarArgs(rd=11, rs1=4, imm=1024)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=12, rs1=8, imm=1024)),
        Instruction(mnemonic="dma.store.ch<N>", args=DmaArgs(rd=8, rs1=4, rs2=10, channel=0)),
        Instruction(mnemonic="dma.store.ch<N>", args=DmaArgs(rd=12, rs1=11, rs2=10, channel=1)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=1)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (0x0000, torch.ones((32, 32), dtype=torch.float8_e4m3fn)),
        (0x0400, torch.ones((32, 32), dtype=torch.float8_e4m3fn)),
        (0x0800, torch.ones((32, 32), dtype=torch.float8_e4m3fn)),
    ]

    golden_result: tuple[int, torch.Tensor] = (
        0x0C00,
        torch.cat(
            (
                gemma_mlp_gate_up_forward(
                    torch.ones((32, 32), dtype=torch.float8_e4m3fn),
                    torch.ones((32, 32), dtype=torch.float8_e4m3fn),
                    torch.ones((32, 32), dtype=torch.float8_e4m3fn),
                    use_gelu=False,
                )
                .to(torch.bfloat16)[:, :16],
                gemma_mlp_gate_up_forward(
                    torch.ones((32, 32), dtype=torch.float8_e4m3fn),
                    torch.ones((32, 32), dtype=torch.float8_e4m3fn),
                    torch.ones((32, 32), dtype=torch.float8_e4m3fn),
                    use_gelu=False,
                )
                .to(torch.bfloat16)[:, 16:],
            ),
            dim=0,
        ).contiguous(),
    )
```

## gemma_attention.py

SUMMARY: This document demonstrates how to construct a scaled dot-product attention kernel for an NPU by composing matrix operations (Q@K matmul), vector operations (softmax via exp/sum/reciprocal), and DMA transfers into a statically-scheduled instruction program.

```python
from typing import List, Tuple, Any
import math
import torch
from ...software import Instruction, Program
from npu_model.isa import DmaArgs, MatrixArgs, VectorArgs, ScalarArgs


class GemmaAttentionProgram(Program):
    """
    Gemma attention kernel program (simplified, single-head).

    This program demonstrates a scaled dot-product attention block using
    the NPU ISA:
      - `matmul.mxu0` for Q @ K and softmax(QK^T) @ V
      - `vexp`, `vreduce.sum`, `vrcp`, and `vmul` to implement softmax.
    """

    instructions: List[Instruction[Any]] = [
        # Register setup (VMEM)
        Instruction(mnemonic="lui", args=ScalarArgs(rd=1, imm=0x2)),
        Instruction(mnemonic="lui", args=ScalarArgs(rd=2, imm=0x2)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=2, rs1=2, imm=0x400)),
        
        # Register setup (DRAM)
        Instruction(mnemonic="addi", args=ScalarArgs(rd=6, rs1=0, imm=0x0000)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=7, rs1=0, imm=0x0400)),
        
        # DMA: Load Q, K, scale from DRAM to VMEM
        Instruction(mnemonic="dma.config.ch<N>", args=DmaArgs(rs1=0, channel=0)),
        Instruction(mnemonic="dma.load.ch<N>", args=DmaArgs(rd=1, rs1=6, rs2=11, channel=0)),
        Instruction(mnemonic="dma.load.ch<N>", args=DmaArgs(rd=2, rs1=7, rs2=11, channel=1)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=1)),
        
        # Load vectors from VMEM to MRF
        Instruction(mnemonic="vload", args=VectorArgs(vd=0, rs1=1, imm12=0)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=1, rs1=2, imm12=0)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=2, rs1=4, imm12=0)),
        
        # Matrix multiply: Q @ K
        Instruction(mnemonic="vmatpush.weight.mxu0", args=VectorArgs(vd=0, vs1=1)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=17)),
        Instruction(mnemonic="vmatmul.mxu0", args=MatrixArgs(vd=0, vs1=0, vs2=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=33)),
        Instruction(mnemonic="vmatpop.bf16.acc.mxu0", args=VectorArgs(vd=3, vs1=0)),
        
        # Scale scores
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=4, vs1=3, vs2=2)),
        
        # Softmax computation
        Instruction(mnemonic="vexp.bf16", args=VectorArgs(vd=5, vs1=4)),
        Instruction(mnemonic="vredsum.row.bf16", args=VectorArgs(vd=6, vs1=5)),
        Instruction(mnemonic="vrecip.bf16", args=VectorArgs(vd=7, vs1=6)),
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=8, vs1=5, vs2=7)),
        
        # Store result back to DRAM
        Instruction(mnemonic="vstore", args=VectorArgs(vd=8, rs1=5, imm12=0)),
        Instruction(mnemonic="dma.store.ch<N>", args=DmaArgs(rd=10, rs1=5, rs2=12, channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (0x0000, torch.zeros((32, 32), dtype=torch.float8_e4m3fn)),
        (0x0400, torch.zeros((32, 32), dtype=torch.float8_e4m3fn)),
        (0x0800, torch.full((32, 16), 1.0 / math.sqrt(16.0), dtype=torch.bfloat16)),
    ]
```

## matmul.py

SUMMARY: Demonstrates how to construct a statically-scheduled NPU kernel program for matrix multiplication by composing scalar, vector, and DMA instructions with explicit memory layout management and timing delays.

```python
from typing import List, Tuple, Any
import torch
from npu_model.isa import (
    DmaArgs,
    MatrixArgs,
    ScalarArgs,
    VectorArgs,
)
from ...software import Instruction, Program

class MatmulProgram(Program):
    """
    Rewritten Matmul test using structured Args dataclasses.
    """

    instructions: List[Instruction[Any]] = [
        # Load DRAM base addresses into scalar registers
        Instruction(mnemonic="addi", args=ScalarArgs(rd=4, rs1=0, imm=0x000)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=5, rs1=0, imm=0x400)),
        
        # Load VMEM base addresses using LUI + ADDI
        Instruction(mnemonic="lui", args=ScalarArgs(rd=1, imm=0x2)),
        Instruction(mnemonic="lui", args=ScalarArgs(rd=2, imm=0x2)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=2, rs1=2, imm=0x400)),
        Instruction(mnemonic="lui", args=ScalarArgs(rd=3, imm=0x3)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=3, rs1=3, imm=-2048)),
        
        # Set transfer size
        Instruction(mnemonic="addi", args=ScalarArgs(rd=6, rs1=0, imm=1024)),
        
        # DMA load activation and weights into VMEM
        Instruction(mnemonic="dma.config.ch<N>", args=DmaArgs(rs1=0, channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(mnemonic="dma.load.ch<N>", args=DmaArgs(rd=1, rs1=4, rs2=6, channel=0)),
        Instruction(mnemonic="dma.load.ch<N>", args=DmaArgs(rd=2, rs1=5, rs2=6, channel=1)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=1)),
        Instruction("delay", args=ScalarArgs(imm=16)),
        
        # Load vectors from VMEM into MRF
        Instruction(mnemonic="vload", args=VectorArgs(vd=0, rs1=1, imm12=0)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=1, rs1=2, imm12=0)),
        Instruction("delay", args=ScalarArgs(16)),
        
        # Matrix multiply: push weights, execute matmul, pop result
        Instruction(mnemonic="vmatpush.weight.mxu0", args=VectorArgs(vd=0, vs1=1)),
        Instruction(mnemonic="delay", args=ScalarArgs(16)),
        Instruction(mnemonic="vmatmul.mxu0", args=MatrixArgs(vd=0, vs1=0, vs2=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=32)),
        Instruction(mnemonic="vmatpop.bf16.acc.mxu0", args=VectorArgs(vd=2, vs1=0)),
        
        # Store result back to VMEM
        Instruction(mnemonic="vstore", args=VectorArgs(vd=2, rs1=3, imm12=0)),
        Instruction(mnemonic="vstore", args=VectorArgs(vd=3, rs1=3, imm12=32)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=16)),
        
        # DMA store result to DRAM
        Instruction(mnemonic="lui", args=ScalarArgs(rd=10, imm=1)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=10, rs1=10, imm=-2048)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=11, rs1=10, imm=1024)),
        Instruction(mnemonic="dma.store.ch<N>", args=DmaArgs(rd=10, rs1=3, rs2=6, channel=0)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=12, rs1=3, imm=1024)),
        Instruction(mnemonic="dma.store.ch<N>", args=DmaArgs(rd=11, rs1=12, rs2=6, channel=1)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=1)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (0x0000, torch.eye(32, 32, dtype=torch.float8_e4m3fn)),
        (0x0400, (2 * torch.eye(32, 32, dtype=torch.float32)).to(torch.float8_e4m3fn)),
    ]

    golden_result: Tuple[int, torch.Tensor] = (
        0x0800,
        torch.cat((MATMUL_RESULT[:, :16], MATMUL_RESULT[:, 16:]), dim=0),
    )
```

## gemma_rms_norm.py

SUMMARY: Demonstrates how to translate a PyTorch RMS normalization operation into a statically-scheduled NPU kernel program using vector operations, DMA transfers, and memory layout management for BF16 tensor computations.

```python
from typing import List, Tuple, Any
from ...software import Instruction, Program
import torch
from ...workload.gemma_blocks import gemma_rms_norm_forward
from npu_model.isa import DmaArgs, MatrixArgs, VectorArgs, ScalarArgs


class GemmaRmsNormProgram(Program):
    """
    Gemma RMS norm program.
    RMS norm: x * rsqrt(mean(x^2) + eps).
    Row-wise mean via transpose + vreduce.sum (second-to-last dim) + vbroadcast.cols.
    """

    instructions: List[Instruction[Any]] = [
        # VMEM bases (use LUI+ADDI so immediates stay 12-bit clean)
        Instruction(mnemonic="lui", args=ScalarArgs(rd=1, imm=0x2)),
        Instruction(mnemonic="lui", args=ScalarArgs(rd=2, imm=0x2)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=2, rs1=2, imm=0x400)),
        Instruction(mnemonic="lui", args=ScalarArgs(rd=3, imm=0x3)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=3, rs1=3, imm=-2048)),
        
        # DRAM bases
        Instruction(mnemonic="addi", args=ScalarArgs(rd=4, rs1=0, imm=0x0000)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=5, rs1=0, imm=0x0400)),
        Instruction(mnemonic="lui", args=ScalarArgs(rd=6, imm=0x1)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=6, rs1=6, imm=-2048)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=7, rs1=0, imm=1024)),

        # DRAM -> VMEM
        Instruction(mnemonic="dma.config.ch<N>", args=DmaArgs(rs1=0, channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(mnemonic="dma.load.ch<N>", args=DmaArgs(rd=1, rs1=4, rs2=7, channel=0)),
        Instruction(mnemonic="dma.load.ch<N>", args=DmaArgs(rd=2, rs1=5, rs2=7, channel=1)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=1)),

        # VMEM -> MRF
        Instruction(mnemonic="vload", args=VectorArgs(vd=0, rs1=1, imm12=0)),  # x
        Instruction(mnemonic="vload", args=VectorArgs(vd=1, rs1=2, imm12=0)),  # eps

        # x_sq = x * x
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=2, vs1=0, vs2=0)),
        # sum_sq over columns, broadcast back across each row
        Instruction(mnemonic="vredsum.row.bf16", args=VectorArgs(vd=3, vs1=2)),
        # mean_sq = sum_sq * (1/ROW_SIZE)
        Instruction(mnemonic="vli.all", args=VectorArgs(vd=4, imm=16)),
        Instruction(mnemonic="vrecip.bf16", args=VectorArgs(vd=5, vs1=4)),
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=6, vs1=3, vs2=5)),
        # var_eps = var + eps
        Instruction(mnemonic="vadd.bf16", args=VectorArgs(vd=7, vs1=6, vs2=1)),
        # rsqrt = 1/sqrt(var_eps)
        Instruction(mnemonic="vsqrt.bf16", args=VectorArgs(vd=8, vs1=7)),
        Instruction(mnemonic="vrecip.bf16", args=VectorArgs(vd=9, vs1=8)),
        # output = x * rsqrt
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=10, vs1=0, vs2=9)),

        # MRF -> VMEM -> DRAM
        Instruction(mnemonic="vstore", args=VectorArgs(vd=10, rs1=3, imm12=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=20)),
        Instruction(mnemonic="dma.store.ch<N>", args=DmaArgs(rd=6, rs1=3, rs2=7, channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (0x0000, torch.randn(32, 16, dtype=torch.bfloat16)),
        (0x0400, torch.full((32, 16), 1e-6, dtype=torch.bfloat16)),
    ]

    golden_result: tuple[int, torch.Tensor] = (
        0x0800,
        gemma_rms_norm_forward(torch.randn(32, 16, dtype=torch.bfloat16)).to(torch.bfloat16),
    )
```

## dma_stall.py

SUMMARY: Demonstrates how to construct a DMA-based NPU kernel program with multi-channel loads, stalling/synchronization, vector register operations, and matrix multiplication using the npu_model ISA.

```python
from typing import List, Tuple, Any
from npu_model.software import Instruction, Program
from npu_model.isa import DmaArgs, MatrixArgs, VectorArgs, ScalarArgs
import torch


class DMAStallProgram(Program):
    """
    A simple program demonstrating DMA loads, stalling logic, and matrix multiplication
    updated for the latest npu_model ISA.
    """

    instructions: List[Instruction[Any]] = [
        # Setup Scalar Registers
        Instruction(mnemonic="addi", args=ScalarArgs(rd=1, rs1=0, imm=0)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=2, rs1=0, imm=1024)),
        
        # Configure DMA Channels
        Instruction(mnemonic="dma.config.ch<N>", args=DmaArgs(rs1=0, channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        
        # Load from DRAM to VMEM on multiple channels
        Instruction(
            mnemonic="dma.load.ch<N>", args=DmaArgs(rd=0, rs1=1, rs2=2, channel=0)
        ),
        Instruction(
            mnemonic="dma.load.ch<N>", args=DmaArgs(rd=1, rs1=2, rs2=2, channel=1)
        ),
        
        # Synchronize DMA operations
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=1)),
        
        # Move VMEM data to computational registers
        Instruction(mnemonic="vload", args=VectorArgs(vd=1, rs1=0, imm12=0)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=0, rs1=1, imm12=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=100)),
        
        # Push to MXU weight buffer
        Instruction(mnemonic="vmatpush.weight.mxu0", args=VectorArgs(vd=0, vs1=0)),
        
        # Overlapped DMA loads with computation
        Instruction(
            mnemonic="dma.load.ch<N>", args=DmaArgs(rd=3, rs1=0, rs2=1, channel=0)
        ),
        Instruction(
            mnemonic="dma.load.ch<N>", args=DmaArgs(rd=4, rs1=1, rs2=1, channel=1)
        ),
        
        # Matrix multiplication
        Instruction(mnemonic="vmatmul.mxu0", args=MatrixArgs(vd=0, vs1=1, vs2=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=32)),
        Instruction(mnemonic="vmatpop.fp8.acc.mxu0", args=VectorArgs(vd=0, vs1=0)),
        
        # Wait for overlapped loads to complete
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=1)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=0)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (0, torch.eye(32, 32, dtype=torch.float8_e4m3fn)),
        (1024, torch.eye(32, 32, dtype=torch.float8_e4m3fn)),
    ]
```

## vpu_tests.py

SUMMARY: Demonstrates how to construct a statically-scheduled NPU kernel program using Instruction objects with scalar, vector, and DMA arguments, including memory layout configuration, data transfer orchestration, and vector arithmetic operations on bf16 data.

```python
from typing import List, Tuple, Any
from ...software import Instruction, Program
import torch
from npu_model.isa import (
    DmaArgs,
    ScalarArgs,
    VectorArgs,
)

# Memory layout constants
DRAM_INPUT_BASE = 0x0000
DRAM_OUTPUT_BASE = 0x0400
VMEM_INPUT_BASE = 0x2000
VMEM_OUTPUT_BASE = 0x2400

# Input tensor: one MRF register worth of bf16 data
INPUT = torch.arange(32 * 16, dtype=torch.bfloat16).reshape(32, 16)


class VectorArithmeticProgram(Program):
    """
    Basic arithmetic correctness: demonstrates DMA transfers, vector loads/stores,
    and chained vector operations (add, subtract, multiply).
    """

    instructions: List[Instruction[Any]] = [
        # Initialize base addresses in scalar registers
        Instruction(mnemonic="addi", args=ScalarArgs(rd=1, rs1=0, imm=VMEM_INPUT_BASE)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=2, rs1=0, imm=VMEM_OUTPUT_BASE)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=3, rs1=0, imm=DRAM_INPUT_BASE)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=4, rs1=0, imm=DRAM_OUTPUT_BASE)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=5, rs1=0, imm=1024)),
        
        # DMA: DRAM -> VMEM
        Instruction(mnemonic="dma.config.ch<N>", args=DmaArgs(rs1=0, channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(mnemonic="dma.load.ch<N>", args=DmaArgs(rd=1, rs1=3, rs2=5, channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction("delay", args=ScalarArgs(imm=16)),
        
        # Vector operations: load, add, subtract, multiply
        Instruction(mnemonic="vload", args=VectorArgs(vd=0, rs1=1, imm12=0)),
        Instruction("delay", args=ScalarArgs(imm=16)),
        Instruction(mnemonic="vadd.bf16", args=VectorArgs(vd=1, vs1=0, vs2=0)),
        Instruction("delay", args=ScalarArgs(imm=32)),
        Instruction(mnemonic="vsub.bf16", args=VectorArgs(vd=2, vs1=1, vs2=0)),
        Instruction("delay", args=ScalarArgs(imm=32)),
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=3, vs1=2, vs2=0)),
        Instruction("delay", args=ScalarArgs(imm=32)),
        Instruction(mnemonic="vstore", args=VectorArgs(vd=3, rs1=2, imm12=0)),
        Instruction("delay", args=ScalarArgs(imm=16)),
        
        # DMA: VMEM -> DRAM
        Instruction(mnemonic="dma.store.ch<N>", args=DmaArgs(rd=4, rs1=2, rs2=5, channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_INPUT_BASE, INPUT),
    ]

    golden_result: tuple[int, torch.Tensor] = (
        DRAM_OUTPUT_BASE,
        (INPUT**2),
    )
```
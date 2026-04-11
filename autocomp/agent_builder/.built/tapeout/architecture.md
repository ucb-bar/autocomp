# Hardware Architecture Summary

The Tapeout NPU is a statically-scheduled, single-issue, in-order accelerator with a scalar RISC-V-style control path and a tile-oriented tensor datapath. All execution timing is deterministic and compiler-visible; there is no hardware dependency tracking, speculation, or out-of-order execution. The programmer is responsible for inserting explicit `delay` instructions and ordering operations to respect latencies and structural hazards.

## Programming Model

The machine fetches one 32-bit instruction per cycle from IMEM, decodes it, and issues it to the appropriate functional unit. Long-latency tensor operations (MXU matmuls, VPU ops, XLU transforms) execute concurrently with each other and with scalar instructions, but only one new instruction issues per cycle. Issue stalls when the targeted unit is busy. Control flow uses branches and jumps with exactly 2 mandatory delay slots (no branch-in-delay-slot allowed). The `delay N` instruction stalls the frontend for N cycles. `dma.wait.chN` stalls the frontend until the specified DMA channel completes.

## Architectural State

- **Scalar registers:** 32 × 32-bit (`x0`–`x31`; `x0` hardwired to zero). Standard RV32I ABI.
- **Tensor registers:** 64 × 1024-byte (`m0`–`m63`). Each stores one 32×32 FP8 tile or one 32×16 BF16 half-tile. A full 32×32 BF16 tile spans two consecutive registers (e.g., `m4`/`m5`).
- **Scale registers:** 32 × 8-bit (`e0`–`e31`). Used for FP8 quantization/dequantization scaling.
- **MXU state (per MXU, ×2):** 2 weight slots (1024 bytes each, holding 32×32 FP8 tiles) and 2 accumulation buffers (2048 bytes each, holding 32×32 BF16 tiles). These are not directly addressable as `m` registers; data moves via explicit push/pop instructions.
- **DMA state:** Shared `dma.base` register (32-bit), 8 independent channels each with busy/idle state.
- **PC:** Instruction-word index (increments by 1 per instruction, not by 4).

## Memory Hierarchy

| Region | Size | Access Path | Bus Width | Throughput |
|--------|------|-------------|-----------|------------|
| IMEM | 64 KiB | Instruction fetch only | 32-bit | 1 insn/cycle |
| VMEM | 1 MiB | Scalar load/store, vload/vstore, DMA target | 512-bit | 64 B/cycle (1 beat/cycle) |
| DRAM | 16 GiB | DMA only | 32-bit | 2 B/cycle (4 B every 2 cycles) |

**IMPORTANT: All addresses used in instructions are zero-based offsets, NOT physical SoC addresses.**
VMEM addresses range from `0x0000` to `0xFFFFF` (1 MiB). DRAM addresses range from `0x0000` upward.
For example, to access VMEM offset `0x2000`, load the value `0x2000` into a scalar register — do NOT use `0x2000_0000`.
The test harness places data at small offsets (e.g., DRAM activation at `0x0000`, weight at `0x0400`, output at `0x0800`;
VMEM staging at `0x2000`, `0x2400`, `0x2800`). These are the actual values to put in scalar registers.

**VMEM** is the sole on-chip data memory. All tensor loads/stores and scalar data accesses target VMEM. A 1024-byte tensor register load/store (`vload`/`vstore`) takes 16 cycles. A 2048-byte BF16 accumulator push/pop takes 32 cycles.

**DRAM** is accessible only through DMA. Transfer time is `max(offchip_cycles, vmem_cycles)` where offchip_cycles = `ceil((bytes + 8) / 4) × 2` and vmem_cycles = `ceil(bytes / 64)`. DRAM bandwidth (2 B/cycle) is the dominant bottleneck for large transfers. All DMA addresses and sizes must be 32-byte aligned.

## Compute Units

**MXU0 (Systolic) and MXU1 (Inner-Product Tree):** Each performs 32×32 FP8 matrix multiply with BF16 accumulation. Latency: 32 cycles per matmul. Peak throughput: 2048 FLOPs/cycle each (4096 combined). Operands: activation from tensor register (FP8), weight from local weight slot (FP8), result in local accumulation buffer (BF16). Both MXUs can execute concurrently.

**VPU (16 BF16 lanes):** Operates on 32×16 BF16 half-tiles (one `m` register). Supports elementwise add, sub, mul, min, max, ReLU, reciprocal, exp, exp2, sin, cos, tanh, log2, sqrt, and column/row reductions. Pipelineable ops (add, sub, mul, min, max, mov, relu, reductions): 2-cycle latency. Non-pipelineable ops (exp, exp2, recip, sin, cos, tanh, log2, sqrt): 8-cycle latency.

**XLU (Transform/Reduction):** Transpose (`vtrpose.xlu`), row-reduce max, row-reduce sum. 4-cycle latency per operation.

**Scalar ALU:** Standard RV32I integer arithmetic, branches, scalar VMEM load/store. All scalar memory ops are blocking.

## Key Constraints for Code Optimization

1. **All on-chip data transfers are blocking.** `vload`, `vstore`, `vmatpush.*`, `vmatpop.*` stall the pipeline for their full transfer duration. Minimizing unnecessary data movement is critical.

2. **MXU data staging is explicit.** Weights must be pushed to weight slots (`vmatpush.weight.*`, 16 cycles) before matmul. Accumulator preload (`vmatpush.acc.*`) and spill (`vmatpop.*`) are separate blocking operations. BF16 accumulator push/pop moves 2048 bytes (32 cycles); FP8 moves 1024 bytes (16 cycles).

3. **Overlap is the primary performance lever.** MXU0, MXU1, VPU, XLU, and DMA can all be active simultaneously. Schedule `delay` instructions to cover matmul latency (32 cycles) while issuing other work. Use both MXUs in parallel when possible.

4. **DRAM bandwidth is scarce (2 B/cycle vs. 64 B/cycle VMEM).** Prefetch data via DMA well ahead of use. Use multiple DMA channels (8 available) to overlap transfers. Each channel supports one outstanding transfer; issuing to a busy channel is illegal.

5. **BF16 tile = 2 consecutive `m` registers.** Operations producing or consuming full 32×32 BF16 tiles (matmul results, `vmatpop.bf16.*`, `vmatpush.acc.bf16.*`) use register pairs. VPU operations work on individual half-tiles, so processing a full BF16 tile requires two VPU instructions on consecutive registers.

6. **No hardware hazard detection.** The compiler must insert sufficient `delay` cycles between a producing instruction and any consumer. Matmul: 32 cycles. VPU simple: 2 cycles. VPU complex: 8 cycles. XLU: 4 cycles. Blocking transfers complete before the next instruction issues.

7. **Branch delay slots (2 required).** Two instructions after a branch/jump always execute. Fill with useful work or NOPs. No branches allowed in delay slot positions.

8. **Immediate encoding limits.** Scalar immediates are 12-bit sign-extended. Addresses larger than ±2047 require `lui`/`addi` sequences. VMEM tensor addresses use `imm12 << 5` (32-byte granularity, covering ±64 KiB offset range).
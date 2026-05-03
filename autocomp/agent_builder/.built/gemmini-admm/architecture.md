## Hardware Architecture Summary: Gemmini (4x4 PE configuration)

This agent targets a small Gemmini configuration with a 4x4 PE systolic array, used for the admm-multifunction workload. The architectural model is the same as the larger Gemmini configurations described below; the main differences are tile sizes and the mvout configuration interface.

### Decoupled Access/Execute Architecture

Gemmini is a decoupled access/execute architecture: memory-access and execute instructions happen concurrently in different regions of the hardware. The accelerator exposes three controllers:

- **ExecuteController** — handles `preload` and `compute` instructions on the systolic array.
- **LoadController** — handles `mvin` (DRAM → scratchpad/accumulator).
- **StoreController** — handles `mvout` (scratchpad/accumulator → DRAM).

Gemmini includes a reorder buffer (ROB) which detects hazards between instructions in different controllers. Each controller also handles its own dependencies and hazards internally.

### Memory: Scratchpad and Accumulator

Gemmini's private memory is **row-addressed**: each row is `DIM` elements wide, where `DIM` is the number of PEs across the width of the systolic array (here, `DIM = 4`). Elements are of type `inputType` in the scratchpad, and of type `accType` in the accumulator.

Every private Gemmini memory address is 32 bits long. The three most-significant bits are reserved:

- **Bit 31 (MSB)** is `0` for scratchpad addresses and `1` for accumulator addresses.
- **Bit 30** is ignored for scratchpad addresses or for accumulator reads. For accumulator writes, `0` overwrites the data at that address and `1` accumulates on top of it.
- **Bit 29** is ignored for scratchpad addresses or for accumulator writes. For accumulator reads, `0` reads scaled-down `inputType` data and `1` reads `accType` data. When bit 29 is `1`, no activation function or scaling is applied to the output.

### Optimization Principles for the admm Workload

The admm-multifunction workload performs a sequence of small operations, so the dominant cost is per-instruction overhead and unnecessary cross-controller dependencies. Effective optimizations on this configuration tend to focus on:

- **Removing or merging instructions** rather than re-tiling, since tiles are already small.
- **Hoisting redundant operations out of loops** and propagating constants so the accelerator sees the smallest possible instruction stream.
- **Using the bias path for matrix addition** — `compute_preloaded` and `compute_accumulated` can add a bias matrix `D` for free.
- **Pipelining loads, computes, and stores** across the three controllers to overlap them.
- **Eliminating data dependencies and unnecessary fences** so the ROB can issue instructions out of order.
- **Moving CPU-side computation onto the accelerator** when possible (e.g. expressing scaling via `scale_factor` in `config_ld`/`config_st`).

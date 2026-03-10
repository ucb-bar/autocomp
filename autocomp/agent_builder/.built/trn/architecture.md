# AWS Trainium 1 (NeuronCore-v2) Architecture Summary

AWS Trainium 1 (trn1 instances) uses NeuronCore-v2, the second-generation Neuron accelerator. NKI (Neuron Kernel Interface) targets NeuronCore-v2 and NeuronCore-v3 (trn2/inf2) but not NeuronCore-v1 (inf1). Each NeuronCore is an independent compute unit; a trn1.32xlarge has 32 NeuronCores with 16 GB of HBM per NeuronCore (512 GB total device memory).

## Programming Model

All operations execute on statically-shaped tensors — tensor dimensions must be known at compile time. NKI kernels are compiled ahead of time through the NeuronX compiler (`neuronx-cc`), which produces NEFF binaries. The programming model is single-kernel: each NKI kernel runs on one NeuronCore with explicit control over data movement and compute.

## Memory Hierarchy

The NeuronCore-v2 has a multi-level memory hierarchy:

- **HBM (High Bandwidth Memory)**: Off-chip, 16 GB per NeuronCore. Serves as the primary storage for model weights, activations, and KV caches. Data must be explicitly moved between HBM and on-chip memories via DMA.
- **SBUF (State Buffer)**: On-chip SRAM used for input operands. Data is loaded from HBM into SBUF before compute.
- **PSUM (Partial Sum Buffer)**: On-chip SRAM used for accumulation results, particularly from the matrix multiplication engine.

Data movement between HBM and on-chip memories (SBUF/PSUM) is a primary performance bottleneck. The compiler includes specific optimizations to minimize data transfers (e.g., the `--model-type unet-inference` flag exists specifically to reduce "excessive performance-impacting data transfers"). Weight loading into NeuronCore memory is a significant cost, as evidenced by the `--enable-fast-context-switch` option that defers weight loading.

## Compute Units

Each NeuronCore contains distinct compute engines:

- **Matrix Multiplication (Matmult) Engine**: Dedicated hardware for matrix multiply operations. This is the primary throughput engine and is architecturally distinct from other compute paths. The compiler can apply different precision settings specifically to matmult-engine operations versus other operations. The matmult engine can also be repurposed for tensor transpose operations (`fast-relayout`), trading bit-accuracy for performance.
- **Vector Engine**: Handles element-wise and vector operations.
- **Scalar Engine**: Handles scalar computations.

## Data Types and Precision

- **Natively supported compute types**: BF16, FP16
- **FP32**: Accepted at the interface level but automatically cast to BF16 or FP16 internally. The matmult engine and other compute paths can use different cast settings independently.
- **INT8 (s8)**: Supported for weight storage, reducing memory bandwidth requirements. Weights are dequantized to BF16/FP16 for computation.
- **FP8 (e4m3)**: Supported as an auto-cast target (4-bit exponent, 3-bit mantissa).
- **TF32 (TensorFloat-32)**: Supported as an auto-cast target.
- **Accumulation precision**: By default, accumulations (notably in softmax and layernorm) occur in the operand's precision. The `--enable-mixed-precision-accumulation` flag forces FP32 intermediate accumulation with cast-back, indicating the hardware can accumulate in FP32 when explicitly requested.

## Key Constraints and Optimization Considerations

1. **Static shapes**: All tensor dimensions are fixed at compile time. Variable-length inputs require bucketing (compiling multiple kernels for different size buckets).

2. **Data movement dominance**: Transfers between HBM and on-chip SRAM are the primary performance bottleneck. Minimizing data movement and maximizing data reuse in on-chip memory is critical.

3. **Numerical edge cases on trn1**: Compiler-introduced matrix-multiply transpose operations can generate infinity values that propagate to NaN. The `--enable-saturate-infinity` flag (which converts ±infinity to MAX/MIN_FLOAT) addresses this but incurs a performance penalty. This issue is specific to trn1 — trn2 handles it in hardware.

4. **Operator size limits**: There are size limits on operators that can be mapped to hardware. Normalization operators can exceed these limits, requiring compiler intervention.

5. **BF16 as default precision**: BF16 is the default auto-cast target, chosen for its balance of performance and dynamic range preservation. Casting inputs to BF16/FP16 before compilation reduces tensor transfer overhead compared to FP32.

6. **Bandwidth optimization via quantization**: INT8 weight storage specifically targets memory bandwidth reduction, confirming bandwidth as a key constraint.
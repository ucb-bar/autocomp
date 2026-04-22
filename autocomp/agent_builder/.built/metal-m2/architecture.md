## Hardware Architecture Summary

The Apple M2 GPU is a unified-memory mobile/desktop GPU with 8 cores, belonging to the Apple8 GPU family. It runs Metal Shading Language 4.0 compute kernels. Because the M2 uses a unified memory architecture, the CPU and GPU share the same physical DRAM — there are no discrete PCIe transfers, and buffers allocated with `MTLResourceStorageModeShared` are directly accessible by both processors without explicit copies. Total recommended working set is approximately 12 GB, with a maximum single buffer allocation of roughly 9 GB.

### Execution Model

The fundamental execution unit is the SIMD-group (wavefront): 32 threads that execute in lockstep on a single compute unit. Threads are organized into threadgroups, which are the unit of scheduling onto a GPU core. A threadgroup can contain up to 1024 threads, with each dimension independently up to 1024 but the product clamped to 1024. Threadgroup dimensions should always be chosen so the total thread count is a multiple of 32 to avoid partial SIMD-groups that waste lanes. Within a threadgroup, SIMD-groups are linear and one-dimensional; the number of SIMD-groups equals `ceil(threads_per_threadgroup / 32)`. SIMD-groups execute concurrently within a threadgroup and make independent forward progress relative to each other unless synchronized by a threadgroup barrier.

SIMD-group intrinsics (shuffle, broadcast, reduction, prefix-sum, ballot, and related operations) allow threads within a 32-wide SIMD-group to exchange data at register speed without touching memory. These are the fastest cross-thread communication mechanism available and should be preferred over threadgroup memory whenever the communication pattern fits within a single SIMD-group. Quad-groups (2×2 subgroups of 4 threads) are also available but are primarily relevant to fragment shading; in compute kernels, SIMD-group operations are the primary tool.

### Memory Hierarchy

**Registers (per-thread private memory):** Each thread has its own register file. Register pressure — the number of live variables a thread holds simultaneously — directly affects occupancy. More registers per thread means fewer concurrent threads (and thus fewer concurrent SIMD-groups) a core can host. Minimizing live variables, avoiding large local arrays, and reusing temporals reduces register pressure and improves occupancy.

**Threadgroup memory (on-chip shared memory):** Each threadgroup has access to 32 KB of fast on-chip scratchpad memory, shared among all threads in the threadgroup. This memory is banked; when all 32 threads in a SIMD-group access consecutive 32-bit-aligned addresses (one element per thread, sequential), the accesses are conflict-free and service in a single transaction. Bank conflicts (multiple threads hitting the same bank with different addresses) serialize and degrade throughput. Threadgroup memory is the primary mechanism for inter-SIMD-group communication within a threadgroup and for tiling strategies that reuse data loaded from device memory. Allocating more than 32 KB per threadgroup causes pipeline creation failure. Larger per-threadgroup shared memory allocations also reduce the number of threadgroups that can be resident simultaneously on a core, reducing occupancy.

**Device memory (off-chip unified DRAM):** This is the main memory pool shared with the CPU. It is high-capacity but high-latency relative to on-chip memories. Bandwidth is shared between CPU and GPU, so minimizing unnecessary CPU memory traffic during GPU-intensive work improves effective GPU bandwidth. Coalesced access patterns — where the 32 threads of a SIMD-group read or write consecutive addresses — merge into fewer memory transactions and are critical for saturating bandwidth. Scattered or strided accesses waste transaction capacity and reduce effective throughput.

**Caches:** The M2 GPU has internal cache hierarchies (L1/L2) that are not explicitly programmable but benefit from spatial and temporal locality. Repeated access to the same or nearby device-memory addresses within a short time window will hit caches. Tiling strategies that load a working set into threadgroup memory and reuse it across many operations effectively bypass cache pressure on device memory.

### Key Optimization Constraints

**Occupancy** is governed by three resources: threadgroup size, registers per thread, and threadgroup memory per threadgroup. A core can host multiple threadgroups concurrently if resources permit. Higher occupancy helps hide memory latency through warp-level multithreading. The practical tradeoff is that maximizing threadgroup size to 1024 is not always optimal — smaller threadgroups (e.g., 256) may achieve higher occupancy if register or shared memory pressure is the bottleneck.

**Threadgroup sizing:** Always a multiple of 32. For 1D work: 32, 64, 128, 256, 512, or 1024. For 2D work: dimensions whose product is a multiple of 32 (e.g., 16×16=256, 8×32=256, 32×32=1024). The choice depends on the data access pattern and shared memory requirements.

**Threadgroup memory budget:** 32 KB hard limit. Tiling strategies must partition work so that the tile's shared data fits within this budget. The 32 KB is shared across all threads in the threadgroup, so per-thread shared allocations scale with threadgroup size.

**SIMD-group operations vs. threadgroup memory:** SIMD-group shuffles and reductions operate at register speed and require no memory allocation or barrier synchronization. They should be used for intra-SIMD-group communication (reductions, scans, data sharing among 32 threads). Threadgroup memory and barriers are needed only for communication across SIMD-groups within a threadgroup.

**Barrier cost:** `threadgroup_barrier(mem_flags::mem_threadgroup)` synchronizes all SIMD-groups in a threadgroup and ensures threadgroup memory visibility. It is necessary for correctness when SIMD-groups share data through threadgroup memory, but each barrier is a synchronization point that can limit instruction-level parallelism. Minimize the number of barriers by restructuring algorithms to batch shared-memory reads and writes.

**No tensor/matrix coprocessor API:** The M2 does not support the Metal tensor API. Matrix operations must be implemented manually using SIMD-group matrix functions (`simdgroup_matrix`) or explicit tiled algorithms over threadgroup memory.
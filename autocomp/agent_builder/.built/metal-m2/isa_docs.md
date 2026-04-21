## Kernel Declaration and Configuration

### [[kernel]]

5.1.3
Compute Functions (Kernels)

A compute function (also called a kernel) is a data-parallel function that is executed over a 1-, 2-, or 3D grid. The following example shows the syntax for declaring a compute function with the kernel or since Metal 2.3 [[kernel]] attribute:

```metal
[[kernel]]
void my_kernel(…) {…}

kernel
void my_kernel2(…) {…}
```

---

### [[max_total_threads_per_threadgroup]]

You can use the [[max_total_threads_per_threadgroup]] function attribute with a kernel function to specify the maximum threads per threadgroup. The value must fit within 32 bits.

Below is an example of a kernel function that uses this attribute:

```metal
[[max_total_threads_per_threadgroup(x)]]
kernel void
my_kernel(…)
{…}
```

If the [[max_total_threads_per_threadgroup]] value is greater than the [MTLDevice maxThreadsPerThreadgroup] property, then compute pipeline state creation fails.

---

### [[required_threads_per_threadgroup]]

In Metal 4 and later, you can use the [[required_threads_per_threadgroup]] function attribute with a kernel function to specify the number of threads per threadgroup. The value must fit within 32 bits. If the [[required_threads_per_threadgroup]] value is set and the [MTLDevice requiredThreadsPerThreadgroup] property is set, the values must be the same; otherwise, the compute pipeline state creation fails.

---

### -fmetal-math-fp32-functions

```
-fmetal-math-fp32-functions=<fast|precise>
```

This option sets the single-precision floating-point math functions described in section 6.5 to call either the fast or precise version. The default is fast. For Apple silicon, starting with Apple GPU Family 4, the math functions honor INF and NaN.

---

### -fmetal-math-mode

```
-fmetal-math-mode=<fast, relaxed, safe>
```

This option sets how aggressive the compiler can be with floating-point optimizations. The default is fast.

If you set the option to fast, it lets the compiler make aggressive, potentially lossy assumptions about floating-point math. These include no NaNs, no INFs, no signed zeros, allow reciprocal, allow reassociation, and FP contract to be fast.

If you set the option to relaxed, it lets the compiler make aggressive, potentially lossy assumptions about floating-point math, but honors INFs and NaNs. These include no signed zeros, allow reciprocal, allow reassociation, and FP contract to be fast. This supports Apple silicon.

## Address Spaces

### device address space

4.1 Device Address Space

The device address space name refers to buffer memory objects allocated from the device memory pool that are both readable and writeable.

---

### constant address space

4.2 Constant Address Space

The constant address space name refers to buffer memory objects allocated from the device memory pool that are read-only. You must declare variables in program scope in the constant address space and initialize them during the declaration statement. The initializer(s) expression must be a core constant expression. (Refer to section 5.20 of the C++17 specification.) The compiler may evaluate a core constant expression at compile time. Variables in program scope have the same lifetime as the program, and their values persist between calls to any of the compute or graphics functions in the program. Pointers or references to the constant address space are allowed as arguments to functions. Writing to variables declared in the constant address space is a compile-time error. Declaring such a variable without initialization is also a compile-time error.

Buffers in the constant address space passed to kernel, vertex, and fragment functions have minimum alignment requirements based on the GPU.

```metal
constant float samples[] = { 1.0f, 2.0f, 3.0f, 4.0f };
```

---

### Thread Address Space

4.3 Thread Address Space

The thread address space refers to the per-thread memory address space. Variables allocated in this address space are not visible to other threads. Variables declared inside a graphics or kernel function are allocated in the thread address space.

```metal
[[kernel]] void
my_kernel(…)
{
    // A float allocated in the per-thread address space
    float x;

    // A pointer to variable x in per-thread address space
    thread float * p = &x;
    …
}
```

---

### Threadgroup Address Space

4.4 Threadgroup Address Space

A GPU compute unit can execute multiple threads concurrently in a threadgroup, and a GPU can execute a separate threadgroup for each of its compute units.

Threads in a threadgroup can work together by sharing data in threadgroup memory, which is faster on most devices than sharing data in device memory. Use the threadgroup address space to:
- Allocate a threadgroup variable in a kernel, mesh, or object function.
- Define a kernel, fragment, or object function parameter that's a pointer to a threadgroup address.

See the Metal Feature Set Tables to learn which GPUs support threadgroup space arguments for fragment shaders.

Threadgroup variables in a kernel, mesh, or object function only exist for the lifetime of the threadgroup that executes the kernel. Threadgroup variables in a mid-render kernel function are persistent across mid-render and fragment kernel functions over a tile.

This example kernel demonstrates how to declare both variables and arguments in the threadgroup address space. (The [[threadgroup]] attribute in the code below is explained in section 5.2.1.)

```metal
kernel void
my_kernel(threadgroup float *sharedParameter [[threadgroup(0)]],
          …)
{
    // Allocate a float in the threadgroup address space.
    threadgroup float sharedFloat;
```

---

### threadgroup address space with [[threadgroup(0)]] attribute

```metal
    // Allocate an array of 10 floats in the threadgroup address space.
    threadgroup float sharedFloatArray[10];
    ...
}
```

For more information about the [[threadgroup(0)]] attribute, see section 5.2.1.

## Thread Identification and Positioning

### thread_position_in_grid

`thread_position_in_grid`
**Availability:** All OS: Metal 1 and later
**Types:** ushort, ushort2, ushort3, uint, uint2, or uint3
**Description:** The thread's position in an N-dimensional grid of threads.

Notes on kernel function attributes:
- The type for declaring [[thread_position_in_grid]], [[threads_per_grid]], [[thread_position_in_threadgroup]], [[threads_per_threadgroup]], [[threadgroup_position_in_grid]], [[dispatch_threads_per_threadgroup]], and [[threadgroups_per_grid]] needs to be a scalar type or a vector type. If it is a vector type, the number of components for the vector types for declaring these arguments need to match.
- The data types for declaring [[thread_position_in_grid]] and [[threads_per_grid]] need to match.
- The data types for declaring [[thread_position_in_threadgroup]], [[threads_per_threadgroup]], and [[dispatch_threads_per_threadgroup]] need to match.

---

### thread_index_in_threadgroup

`thread_index_in_threadgroup`
**Availability:** All OS: Metal 1 and later
**Types:** ushort or uint
**Description:** The unique scalar index of a thread within a threadgroup.

If [[thread_position_in_threadgroup]] is type uint, uint2, or uint3, [[thread_index_in_threadgroup]] needs to be type uint.

---

### thread_position_in_threadgroup

`thread_position_in_threadgroup`
**Availability:** All OS: Metal 1 and later
**Types:** ushort, ushort2, ushort3, uint, uint2, or uint3
**Description:** The thread's unique position within a threadgroup.

Threads are assigned a unique position within a threadgroup (referred to as thread_position_in_threadgroup). The unique scalar index of a thread within a threadgroup is given by thread_index_in_threadgroup.

---

### thread_index_in_simdgroup

`thread_index_in_simdgroup`
**Availability:** macOS: Metal 2 and later / iOS: Metal 2.2 and later
**Types:** ushort or uint
**Description:** The scalar index of a thread within a SIMD-group.

SIMD-groups execute concurrently within a given threadgroup and make independent forward progress with respect to each other, in the absence of threadgroup barrier operations. The thread index in a SIMD-group (given by [[thread_index_in_simdgroup]]) is a value between 0 and SIMD-group size - 1, inclusive. Similarly, the thread index in a quad-group (given by [[thread_index_in_quadgroup]]) is a value between 0 and 3, inclusive.

---

### thread_index_in_quadgroup

`thread_index_in_quadgroup`
**Availability:** macOS: Metal 2.1 and later / iOS: Metal 2 and later
**Types:** ushort or uint
**Description:** The scalar index of a thread within a quad-group.

---

### simdgroup_index_in_threadgroup

`simdgroup_index_in_threadgroup`
**Availability:** macOS: Metal 2 and later / iOS: Metal 2.2 and later
**Types:** ushort or uint
**Description:** The scalar index of a SIMD-group within a threadgroup.

---

### quadgroup_index_in_threadgroup

`quadgroup_index_in_threadgroup`
**Availability:** macOS: Metal 2.1 and later / iOS: Metal 2 and later
**Types:** ushort or uint
**Description:** The scalar index of a quad-group within a threadgroup.

---

### threadgroup_position_in_grid

`threadgroup_position_in_grid`
**Availability:** All OS: Metal 1 and later
**Types:** ushort, ushort2, ushort3, uint, uint2, or uint3
**Description:** The threadgroup's unique position within a grid.

Threadgroups are assigned a unique position within the grid (referred to as threadgroup_position_in_grid).

---

### grid_origin

`grid_origin`
**Availability:** All OS: Metal 1.2 and later
**Types:** ushort, ushort2, ushort3, uint, uint2, or uint3
**Description:** The origin (offset) of the grid over which compute threads that read per-thread stage-in data are launched.

---

### grid_size

`grid_size`
**Availability:** All OS: Metal 1.2 and later
**Types:** ushort, ushort2, ushort3, uint, uint2, or uint3
**Description:** The maximum size of the grid over which compute threads that read per-thread stage-in data are launched.

## Grid and Threadgroup Dimensions

### threads_per_grid

`threads_per_grid`
**Availability:** All OS: Metal 1 and later
**Types:** ushort, ushort2, ushort3, uint, uint2, or uint3
**Description:** The grid size.

The grid size (threads_per_grid) is: (Gx, Gy) = (Wx * Sx, Wy * Sy)

---

### threads_per_threadgroup

`threads_per_threadgroup`
**Availability:** All OS: Metal 1 and later
**Types:** ushort, ushort2, ushort3, uint, uint2, or uint3
**Description:** The thread execution width of a threadgroup.

In Metal 2 and later, the number of threads in the grid does not have to be a multiple of the number of threads in a threadgroup. It is therefore possible that the actual threadgroup size of a specific threadgroup may be smaller than the threadgroup size specified in the dispatch. The [[threads_per_threadgroup]] attribute specifies the actual threadgroup size for a given threadgroup executing the kernel. The [[dispatch_threads_per_threadgroup]] attribute is the threadgroup size specified at dispatch.

---

### threadgroups_per_grid

`threadgroups_per_grid`
**Availability:** All OS: Metal 1 and later
**Types:** ushort, ushort2, ushort3, uint, uint2, or uint3
**Description:** The number of threadgroups in a grid.

---

### threads_per_simdgroup

`threads_per_simdgroup`
**Availability:** macOS: Metal 2 and later / iOS: Metal 2.2 and later
**Types:** ushort or uint
**Description:** The thread execution width of a SIMD-group (compute unit).

The execution width of the compute unit, referred to as the threads_per_simdgroup, determines the recommended size of this smaller group. For best performance, make the total number of threads in the threadgroup a multiple of the threads_per_simdgroup.

---

### thread_execution_width

`thread_execution_width`
**Availability:** All OS: Metal 1 and later [Deprecated as of Metal 3 -- use threads_per_simdgroup]
**Types:** ushort or uint
**Description:** The thread execution width of a SIMD-group (compute unit).

---

### simdgroups_per_threadgroup

`simdgroups_per_threadgroup`
**Availability:** macOS: Metal 2 and later / iOS: Metal 2.2 and later
**Types:** ushort or uint
**Description:** The SIMD-group execution width of a threadgroup.

For standard Metal compute functions (other than tile functions), SIMD-groups are linear and one-dimensional. (Threadgroups may be multidimensional.) The number of SIMD-groups in a threadgroup ([[simdgroups_per_threadgroup]]) is the total number of threads in the threadgroup ([[threads_per_threadgroup]]) divided by the SIMD-group size ([[threads_per_simdgroup]]):

```
simdgroups_per_threadgroup = ceil(threads_per_threadgroup/threads_per_simdgroup)
```

---

### quadgroups_per_threadgroup

`quadgroups_per_threadgroup`
**Availability:** macOS: Metal 2.1 and later / iOS: Metal 2 and later
**Types:** ushort or uint
**Description:** The quad-group execution width of a threadgroup.

The number of quad-groups in a threadgroup (quadgroups_per_threadgroup) is the total number of threads in threadgroup divided by 4, which is the thread execution width of a quad-group:

```
quadgroups_per_threadgroup = ceil(threads_per_threadgroup/4)
```

---

### dispatch_threads_per_threadgroup

`dispatch_threads_per_threadgroup`
**Availability:** All OS: Metal 1 and later
**Types:** ushort, ushort2, ushort3, uint, uint2, or uint3
**Description:** The thread execution width of a threadgroup for threads specified at dispatch.

---

### dispatch_simdgroups_per_threadgroup

`dispatch_simdgroups_per_threadgroup`
**Availability:** macOS: Metal 2 and later / iOS: Metal 2.2 and later
**Types:** ushort or uint
**Description:** The SIMD-group execution width of a threadgroup specified at dispatch.

[[dispatch_simdgroups_per_threadgroup]] and [[dispatch_quadgroups_per_threadgroup]] are similarly computed for threads specified at dispatch.

---

### dispatch_quadgroups_per_threadgroup

`dispatch_quadgroups_per_threadgroup`
**Availability:** macOS: Metal 2.1 and later / iOS: Metal 2 and later
**Types:** ushort or uint
**Description:** The quad-group execution width of a threadgroup specified at dispatch.

---

### SIMD-Groups and Quad-Groups

4.4.1
SIMD-Groups and Quad-Groups

macOS: Metal 2 and later support SIMD-group functions. Metal 2.1 and later support quad-group functions.
iOS: Metal 2.2 and later support some SIMD-group functions. Metal 2 and later support quad-group functions.

Within a threadgroup, you can divide threads into SIMD-groups, which are collections of threads that execute concurrently. The mapping to SIMD-groups is invariant for the duration of a kernel's execution, across dispatches of a given kernel with the same launch parameters, and from one threadgroup to another within the dispatch (excluding the trailing edge threadgroups in the presence of nonuniform threadgroup sizes). In addition, all SIMD-groups within a threadgroup needs to be the same size, apart from the SIMD-group with the maximum index, which may be smaller, if the size of the threadgroup is not evenly divisible by the size of the SIMD-groups.

A quad-group is a SIMD-group with the thread execution width of 4.

For more about kernel function attributes for SIMD-groups and quad-groups, see section 5.2.3.6. For more about threads and thread synchronization, see section 6.9 and its subsections:
- For more about thread synchronization functions, including a SIMD-group barrier, see section 6.9.1.
- For more about SIMD-group functions, see section 6.9.2.
- For more about quad-group functions, see section 6.9.3.

## Synchronization and Barriers

### threadgroup_barrier

```metal
void threadgroup_barrier(mem_flags flags)
```

All threads in a threadgroup executing the kernel, fragment, mesh, or object need to execute this function before any thread can continue execution beyond the threadgroup_barrier.

---

### simdgroup_barrier

```metal
void simdgroup_barrier(mem_flags flags)
```

**Availability:** macOS: Metal 2 and later / iOS: Metal 1.2 and later

All threads in a SIMD-group executing the kernel, fragment, mesh, or object need to execute this function before any thread can continue execution beyond the simdgroup_barrier.

---

### threadgroup_barrier / simdgroup_barrier

If threadgroup_barrier (or simdgroup_barrier) is inside a conditional statement and if any thread enters the conditional statement and executes the barrier function, then all threads in the threadgroup (or SIMD-group) need to execute the barrier function before any threads continue execution beyond the barrier function.

The threadgroup_barrier (or simdgroup_barrier) function can also queue a memory fence (for reads and writes) to ensure the correct ordering of memory operations to threadgroup or device memory.

Table 6.13 describes the bit field values for the mem_flags argument to threadgroup_barrier and simdgroup_barrier. The mem_flags argument ensures the correct memory is in the correct order between threads in the threadgroup or SIMD-group (for threadgroup_barrier or simdgroup_barrier), respectively.

---

### mem_flags enumeration

Table 6.13. Memory flag enumeration values for barrier functions

| Memory flags (mem_flags) | Description |
|---|---|
| `mem_none` | The flag sets threadgroup_barrier or simdgroup_barrier to only act as an execution barrier and doesn't apply a Memory fence. |
| `mem_device` | The flag ensures the GPU correctly orders the memory operations to device memory for threads in the threadgroup or SIMD-group. |
| `mem_threadgroup` | The flag ensures the GPU correctly orders the memory operations to threadgroup memory for threads in a threadgroup or SIMD-group. |
| `mem_texture` | macOS: Metal 1.2 and later / iOS: Metal 2 and later. The flag ensures the GPU correctly orders the memory operations to texture memory for threads in a threadgroup or SIMD-group for a texture with the read_write access qualifier. |
| `mem_threadgroup_imageblock` | The flag ensures the GPU correctly orders the memory operations to threadgroup imageblock memory for threads in a threadgroup or SIMD-group. |
| `mem_object_data` | The flag ensures the GPU correctly orders the memory operations to object_data memory for threads in the threadgroup or SIMD-group. |

## Atomic Operations

### atomic_load_explicit

```metal
C atomic_load_explicit(const volatile threadgroup A* object,
                       memory_order order) // All OS: Since Metal 1.

C atomic_load_explicit(const threadgroup A* object,
                       memory_order order) // All OS: Since Metal 2.
```

---

### atomic_store_explicit

```metal
void atomic_store_explicit(volatile threadgroup A* object,
                           C desired,
                           memory_order order) // All OS: Since Metal 1.

void atomic_store_explicit(volatile device A* object, C desired,
                           memory_order order) // All OS: Since Metal 1.

void atomic_store_explicit(threadgroup A* object, C desired,
                           memory_order order) // All OS: Since Metal 2.

void atomic_store_explicit(device A* object, C desired,
                           memory_order order) // All OS: Since Metal 2.
```

---

### atomic_compare_exchange_weak_explicit

All OS: Support for the atomic_compare_exchange_weak_explicit function supported as indicated; support for memory_order_relaxed for indicating success and failure. If the comparison is true, the value of success affects memory access, and if the comparison is false, the value of failure affects memory access.

```metal
bool atomic_compare_exchange_weak_explicit(threadgroup A* object,
                    C *expected, C desired, memory_order success,
                    memory_order failure) // All OS: Since Metal 2.

bool atomic_compare_exchange_weak_explicit(volatile threadgroup A* object,
                    C *expected, C desired, memory_order success,
                    memory_order failure)  // All OS: Since Metal 1.

bool atomic_compare_exchange_weak_explicit(device A* object,
                    C *expected, C desired, memory_order success,
                    memory_order failure)  // All OS: Since Metal 2.

bool atomic_compare_exchange_weak_explicit(volatile device A* object,
                    C *expected, C desired, memory_order success,
                    memory_order failure)  // All OS: Since Metal 1.
```

---

### atomic_fetch_key_explicit

6.15.4.5 Atomic Fetch and Modify Functions

All OS: The following atomic fetch and modify functions are supported, as indicated. The only supported value for order is memory_order_relaxed.

```metal
C atomic_fetch_key_explicit(threadgroup A* object,
                            M operand,
                            memory_order order) // All OS: Since Metal 2.

C atomic_fetch_key_explicit(volatile threadgroup A* object,
                            M operand,
                            memory_order order) // All OS: Since Metal 1.

C atomic_fetch_key_explicit(device A* object,
                            M operand,
                            memory_order order) // All OS: Since Metal 2.

C atomic_fetch_key_explicit(volatile device A* object,
                            M operand,
                            memory_order order) // All OS: Since Metal 1.
```

The key in the function name is a placeholder for an operation name listed in the first column of Table 6.25, such as atomic_fetch_add_explicit. The operations detailed in Table 6.25 are arithmetic and bitwise computations. The function atomically replaces the value pointed to by object with the result of the specified computation (third column of Table 6.25). The function returns the value that object held previously. There are no undefined results.

These functions are applicable to any atomic object of type atomic_int, and atomic_uint. Atomic add and sub support atomic_float only in device memory.

---

### atomic_key_explicit

```metal
void atomic_key_explicit(device A* object,
                         M operand,
                         memory_order order)

void atomic_key_explicit(volatile device A* object,
                         M operand,
                         memory_order order)
```

The key in the function name is a placeholder for an operation name listed in the first column of Table 6.26, such as atomic_max_explicit. The operations detailed in Table 6.26 are arithmetic. The function atomically replaces the value pointed to by object with the result of the specified computation (third column of Table 6.26). The function returns void. There are no undefined results.

---

### Atomic operations

Table 6.25. Atomic operations

| Key | Operator | Computation |
|---|---|---|
| add | `+` | Addition |
| and | `&` | Bitwise and |
| max | `max` | Compute max |
| min | `min` | Compute min |
| or | `\|` | Bitwise inclusive or |
| sub | `-` | Subtraction |
| xor | `^` | Bitwise exclusive or |

These operations are atomic read-modify-write operations. For signed integer types, the arithmetic operation uses two's complement representation with silent wrap-around on overflow.

---

### memory_order

The enumeration memory_order specifies the detailed regular (nonatomic) memory synchronization operations (see section 29.3 of the C++17 specification) and may provide for operation ordering:

```metal
enum memory_order {
    memory_order_relaxed,
    memory_order_seq_cst
};
```

For atomic operations other than atomic_thread_fence, memory_order_relaxed is the only enumeration value. With memory_order_relaxed, there are no synchronization or ordering constraints; the operation only requires atomicity. These operations do not order memory, but they guarantee atomicity and modification order consistency. A typical use for relaxed memory ordering is updating counters, such as reference counters because this only requires atomicity, but neither ordering nor synchronization.

In Metal 3.2 and later, you can use memory_order_seq_cst on atomic_thread_fence to indicate that everything that happens before a store operation in one thread becomes a visible side effect in the thread that performs the load, and establishes a single total modification order of all tagged atomic operations.

---

### thread_scope

The enumeration thread_scope denotes a set of threads for the memory order constraint that the memory_order provides:

```metal
enum thread_scope {
    thread_scope_thread,
    thread_scope_simdgroup,
    thread_scope_threadgroup,
};
```

## SIMD-Group Voting and Masking

### simd_vote

```metal
class simd_vote {
public:
    explicit constexpr simd_vote(vote_t v = 0);
    explicit constexpr operator vote_t() const;
};
```

---

### simd_active_threads_mask()

```metal
simd_vote simd_active_threads_mask()
```

**Availability:** macOS: Metal 2.1 and later / iOS: Metal 2.2 and later

Returns a simd_vote mask that represents the active threads. This function is equivalent to simd_ballot(true) and sets the bits that represent active threads to 1, and inactive threads to 0.

---

### simd_all(bool expr)

```metal
bool simd_all(bool expr)
```

**Availability:** macOS: Metal 2.1 and later / iOS: Metal 2.2 and later

Returns true if all active threads evaluate expr to true.

---

### simd_any(bool expr)

```metal
bool simd_any(bool expr)
```

**Availability:** macOS: Metal 2.1 and later / iOS: Metal 2.2 and later

Returns true if at least one active thread evaluates expr to true.

---

### simd_ballot(bool expr)

```metal
simd_vote simd_ballot(bool expr)
```

**Availability:** macOS: Metal 2.1 and later / iOS: Metal 2.2 and later

Returns a wrapper type -- see the simd_vote example -- around a bitmask of the evaluation of the Boolean expression for all active threads in the SIMD-group for which expr is true. The function sets the bits that correspond to inactive threads to 0.

---

### simd_is_first

```metal
bool simd_is_first()
```

**Availability:** macOS: Metal 2.1 and later / iOS: Metal 2.2 and later

Returns true if the current thread is the first active thread -- the active thread with the smallest index -- in the current SIMD-group; otherwise, false.

## SIMD-Group Broadcast and Shuffle

### SIMD-Group Functions

6.9.2
SIMD-Group Functions

The <metal_simdgroup> header defines the SIMD-group functions for kernel and fragment functions. macOS supports SIMD-group functions in Metal 2 and later, and iOS supports most SIMD-group functions in Metal 2.2 and later. Table 6.14 and Table 6.15 list the SIMD-group functions and their availabilities in iOS and macOS. See the Metal Feature Set Tables to determine which GPUs support each table.

---

### simd_broadcast

```metal
T simd_broadcast(T data, ushort broadcast_lane_id)
```

**Availability:** macOS: Metal 2 and later / iOS: Metal 2.2 and later

Broadcasts data from the thread whose SIMD lane ID is equal to broadcast_lane_id. The specification doesn't define the behavior when broadcast_lane_id isn't a valid SIMD lane ID or isn't the same for all threads in a SIMD-group.

---

### simd_broadcast_first

```metal
T simd_broadcast_first(T data)
```

**Availability:** macOS: Metal 2.1 and later / iOS: Metal 2.2 and later

Broadcasts data from the first active thread -- the active thread with the smallest index -- in the SIMD-group to all active threads.

---

### simd_shuffle

```metal
T simd_shuffle(T data, ushort simd_lane_id)
```

**Availability:** macOS: Metal 2 and later / iOS: Metal 2.2 and later

Returns data from the thread whose SIMD lane ID is simd_lane_id. The simd_lane_id needs to be a valid SIMD lane ID but doesn't have to be the same for all threads in the SIMD-group.

---

### simd_shuffle_up

```metal
T simd_shuffle_up(T data, ushort delta)
```

**Availability:** macOS: Metal 2 and later / iOS: Metal 2.2 and later

Returns data from the thread whose SIMD lane ID is the difference from the caller's SIMD lane ID minus delta. The value of delta needs to be the same for all threads in a SIMD-group. This function doesn't modify the lower delta lanes of data because it doesn't wrap values around the SIMD-group.

The simd_shuffle_up() function shifts each SIMD-group upward by delta threads. For example, with a delta value of 2, the function:
- Shifts the SIMD lane IDs down by two
- Marks the lower two lanes as invalid

Computed
SIMD lane ID
–2
–1
0
1
2
3
4
5
6
7
8
9
10
11
12 13
valid
0
0
1
1
1
1
1
1
1
1
1
1
1
1
1
1
data
a
b
a
b
c
d
e
f
g
h
i
j
k
l
m
n

The simd_shuffle_up() function is a no-wrapping operation that doesn't affect the lower delta lanes.

---

### simd_shuffle_down

```metal
T simd_shuffle_down(T data, ushort delta)
```

**Availability:** macOS: Metal 2 and later / iOS: Metal 2.2 and later

Returns data from the thread whose SIMD lane ID is the sum of caller's SIMD lane ID and delta. The value for delta needs to be the same for all threads in the SIMD-group. This function doesn't modify the upper delta lanes of data because it doesn't wrap values around the SIMD-group.

Similarly, the simd_shuffle_down() function shifts each SIMD-group downward by the delta threads. Starting with the same initial SIMD-group state, with a delta value of 2, the function:
- Shifts the SIMD lane IDs up by two
- Marks the upper two lanes as invalid

Computed
SIMD lane ID
2
3
4
5
6
7
8
9
10
11
12 13 14 15 16
17
valid 1
1
1
1
1
1
1
1
1
1
1
1
1
1
0
0
data c
d
e
f
g
h
i
j
k
l
m
n
o
p
o
p

The simd_shuffle_down() function is a no-wrapping operation that doesn't affect the upper delta lanes.

---

### simd_shuffle_rotate_up

```metal
T simd_shuffle_rotate_up(T data, ushort delta)
```

**Availability:** macOS: Metal 2.1 and later / iOS: Metal 2.2 and later

Returns data from the thread whose SIMD lane ID is the difference from the caller's SIMD lane ID minus delta. The value of delta needs to be the same for all threads in a SIMD-group. This function wraps values around the SIMD-group.

---

### simd_shuffle_rotate_down

```metal
T simd_shuffle_rotate_down(T data, ushort delta)
```

**Availability:** macOS: Metal 2.1 and later / iOS: Metal 2.2 and later

Returns data from the thread whose SIMD lane ID is the sum of caller's SIMD lane ID and delta. The value for delta needs to be the same for all threads in the SIMD-group. This function wraps values around the SIMD-group.

---

### simd_shuffle_xor

```metal
Ti simd_shuffle_xor(Ti value, ushort mask)
```

**Availability:** macOS: Metal 2 and later / iOS: Metal 2.2 and later

Returns data from the thread whose SIMD lane ID is equal to the bitwise XOR (^) of the caller's SIMD lane ID and mask. The value of mask needs to be the same for all threads in a SIMD-group.

---

### simd_shuffle_and_fill_up

```metal
T simd_shuffle_and_fill_up(T data, T filling_data, ushort delta, ushort modulo)
T simd_shuffle_and_fill_up(T data, T filling_data, ushort delta)
```

**Availability:** All OS: Metal 2.4 and later

Returns data or filling_data from the thread whose SIMD lane ID is the difference from the caller's SIMD lane ID minus delta. If the difference is negative, the operation copies values from the upper delta lanes of filling_data to the lower delta lanes of data. The value of delta needs to be the same for all threads in a SIMD-group. The modulo parameter (when provided) defines the vector width that splits the SIMD-group into separate vectors and must be 2, 4, 8, 16, or 32.

Without the modulo parameter, the function shifts each SIMD-group upward by delta threads -- similar to simd_shuffle_up() -- and assigns the values from the upper filling lanes to the lower data lanes by wrapping the SIMD lane IDs.

With the modulo parameter, the function splits the SIMD-group into vectors, each with size modulo, and shifts each vector by the delta threads. For example, with a modulo value of 8 and a delta value of 2, the function:
- Shifts the SIMD lane IDs down by two
- Assigns the upper two lanes of each vector in filling to the lower two lanes of each vector in data

Computed
SIMD lane ID
–2 –1
0
1
2
3
4
5
–2 –1
0
1
2
3
4
5
data fg fh a
b
c
d
e
f fy fz s
t
u
v
w
x

---

### simd_shuffle_and_fill_down

```metal
T simd_shuffle_and_fill_down(T data, T filling_data, ushort delta, ushort modulo)
T simd_shuffle_and_fill_down(T data, T filling_data, ushort delta)
```

**Availability:** All OS: Metal 2.4 and later

Returns data or filling_data from the thread whose SIMD lane ID is the sum of the caller's SIMD lane ID and delta. If the sum is greater than modulo (or the SIMD-group size if no modulo), the function copies values from the lower delta lanes of filling_data into the upper delta lanes of data. The value of delta needs to be the same for all threads in a SIMD-group. The modulo parameter (when provided) defines the vector width that splits the SIMD-group into separate vectors and must be 2, 4, 8, 16, or 32.

Without the modulo parameter, the function shifts each SIMD-group downward by delta threads -- like simd_shuffle_down() -- and assigns the values from the lower filling lanes to the upper data lanes by wrapping the SIMD lane IDs.

With the modulo parameter, the function splits the SIMD-group into vectors, each with size modulo and shifts each vector by the delta threads. For example, with a modulo value of 8 and a delta value of 2, the function:
- Shifts the SIMD lane IDs up by two
- Assigns the lower two lanes of each vector in filling to the upper two lanes of each vector in data

Computed
SIMD lane ID
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
data
c
d
e
f
g
h fa fb u
v
w
x
y
z fs ft

## SIMD-Group Reductions and Scans

### simd_sum

```metal
T simd_sum(T data)
```

**Availability:** macOS: Metal 2.1 and later / iOS: Metal 2.3 and later

Returns the sum of the input values in data across all active threads in the SIMD-group and broadcasts the result to all active threads in the SIMD-group.

---

### simd_product

```metal
T simd_product(T data)
```

**Availability:** macOS: Metal 2.1 and later / iOS: Metal 2.3 and later

Returns the product of the input values in data across all active threads in the SIMD-group and broadcasts the result to all active threads in the SIMD-group.

---

### simd_min

```metal
T simd_min(T data)
```

**Availability:** macOS: Metal 2.1 and later / iOS: Metal 2.3 and later

Returns data with the lowest value from across all active threads in the SIMD-group and broadcasts that value to all active threads in the SIMD-group.

---

### simd_max

```metal
T simd_max(T data)
```

**Availability:** macOS: Metal 2.1 and later / iOS: Metal 2.3 and later

Returns data with the highest value from across all active threads in the SIMD-group and broadcasts that value to all active threads in the SIMD-group.

---

### simd_and

```metal
Ti simd_and(Ti data)
```

**Availability:** macOS: Metal 2.1 and later / iOS: Metal 2.3 and later

Returns the bitwise AND (&) of data across all active threads in the SIMD-group and broadcasts the result to all active threads in the SIMD-group.

---

### simd_or

```metal
Ti simd_or(Ti data)
```

**Availability:** macOS: Metal 2.1 and later / iOS: Metal 2.3 and later

Returns the bitwise OR (|) of data across all active threads in the SIMD-group and broadcasts the result to all active threads in the SIMD-group.

---

### simd_xor

```metal
Ti simd_xor(Ti data)
```

**Availability:** macOS: Metal 2.1 and later / iOS: Metal 2.3 and later

Returns the bitwise XOR (^) of data across all active threads in the SIMD-group and broadcasts the result to all active threads in the SIMD-group.

---

### simd_prefix_exclusive_sum

```metal
T simd_prefix_exclusive_sum(T data)
```

**Availability:** macOS: Metal 2.1 and later / iOS: Metal 2.3 and later

For a given thread, returns the sum of the input values in data for all active threads with a lower index in the SIMD-group. The first thread in the group, returns T(0).

---

### simd_prefix_exclusive_product

```metal
T simd_prefix_exclusive_product(T data)
```

**Availability:** macOS: Metal 2.1 and later / iOS: Metal 2.3 and later

For a given thread, returns the product of the input values in data for all active threads with a lower index in the SIMD-group. The first thread in the group, returns T(1).

---

### simd_prefix_inclusive_sum

```metal
T simd_prefix_inclusive_sum(T data)
```

**Availability:** macOS: Metal 2.1 and later / iOS: Metal 2.3 and later

For a given thread, returns the sum of the input values in data for all active threads with a lower or the same index in the SIMD-group.

---

### simd_prefix_inclusive_product

```metal
T simd_prefix_inclusive_product(T data)
```

**Availability:** macOS: Metal 2.1 and later / iOS: Metal 2.3 and later

For a given thread, returns the product of the input values in data for all active threads with a lower or the same index in the SIMD-group.

## Quad-Group Shuffle Operations

### quad_shuffle_up

```metal
T quad_shuffle_up(T data, ushort delta)
```

**Availability:** macOS: Metal 2 and later / iOS: Metal 2 and later

Returns data from thread whose quad lane ID is the difference from the caller's quad lane ID minus delta. The value for delta needs to be the same for all threads in a quad-group. This function doesn't modify the lower delta lanes of data because it doesn't wrap values around the quad-group.

The quad_shuffle_up() function shifts each quad-group upward by delta threads. For example, with a delta value of 2, the function:
- Shifts the quad lane IDs down by two
- Marks the lower two lanes as invalid

Computed
quad lane ID
–2
–1
0
1
valid
0
0
1
1
data
a
b
a
b

The quad_shuffle_up() function is a no wrapping operation that doesn't affect the lower delta lanes.

---

### quad_shuffle_down

```metal
T quad_shuffle_down(T data, ushort delta)
```

**Availability:** macOS: Metal 2 and later / iOS: Metal 2 and later

Returns data from the thread whose quad lane ID is the sum of the caller's quad lane ID and delta. The value for delta needs to be the same for all threads in a quad-group. The function doesn't modify the upper delta lanes of data because it doesn't wrap values around the quad-group.

Similarly, quad_shuffle_down() function shifts each quad-group downward by delta threads. Starting with the same initial quad-group state, with a delta of 2, the function:
- Shifts the quad lane IDs up by two
- Marks the upper two lanes as invalid

Computed
quad lane ID
2
3
4
5
valid
1
1
0
0
data
c
d
c
d

The quad_shuffle_down() function is a no wrapping operation that doesn't affect the upper delta lanes.

---

### quad_shuffle_rotate_up

```metal
T quad_shuffle_rotate_up(T data, ushort delta)
```

**Availability:** macOS: Metal 2.1 and later / iOS: Metal 2.2 and later

Returns data from the thread whose quad lane ID is the difference from the caller's quad lane ID minus delta. The value for delta needs to be the same for all threads in a quad-group. This function wraps values around the quad-group.

---

### quad_shuffle_rotate_down

```metal
T quad_shuffle_rotate_down(T data, ushort delta)
```

**Availability:** macOS: Metal 2.1 and later / iOS: Metal 2.2 and later

Returns data from the thread whose quad lane ID is the sum of the caller's quad lane ID and delta. The value for delta needs to be the same for all threads in a quad-group. This function wraps values around the quad-group.

---

### quad_shuffle_xor

```metal
T quad_shuffle_xor(T value, ushort mask)
```

**Availability:** macOS: Metal 2 and later / iOS: Metal 2 and later

Returns data from the thread whose quad lane ID is a bitwise XOR (^) of the caller's quad lane ID and mask. The value of mask needs to be the same for all threads in a quad-group.

---

### quad_shuffle_and_fill_up

```metal
T quad_shuffle_and_fill_up(T data, T filling_data, ushort delta, ushort modulo)
T quad_shuffle_and_fill_up(T data, T filling_data, ushort delta)
```

**Availability:** All OS: Metal 2.4 and later

Returns data or filling_data from the thread whose quad lane ID is the difference from the caller's quad lane ID minus delta. If the difference is negative, the operation copies values from the upper delta lanes of filling_data to the lower delta lanes of data. The value of delta needs to be the same for all threads in a quad-group. The modulo parameter (when provided) defines the width that splits the quad-group into separate vectors and must be 2 or 4.

The quad_shuffle_and_fill_up() function with the modulo parameter splits the quad-group into vectors, each with size modulo and shifts each vector by the delta threads. For example, with a modulo value of 2 and a delta value of 1, the function:
- Shifts the quad lane IDs down by one
- Assigns the upper lane of each vector in filling to the lower lane of each vector in data

Computed
quad lane ID
–1
0
–1
0
data
fb
a
fd
c

Without the modulo parameter, the function shifts each quad-group upward by the delta threads -- similar to quad_shuffle_up() -- and assigns the values from the upper filling lanes to the lower data lanes by wrapping the quad lane IDs.

---

### quad_shuffle_and_fill_down

```metal
T quad_shuffle_and_fill_down(T data, T filling_data, ushort delta, ushort modulo)
```

**Availability:** All OS: Metal 2.4 and later

Returns data or filling_data for each vector, from the thread whose quad lane ID is the sum of caller's quad lane ID and delta. If the sum is greater than the quad-group size, the function copies values from the lower delta lanes of filling_data into the upper delta lanes of data. The value of delta needs to be the same for all threads in a quad-group. The modulo parameter defines the vector width that splits the quad-group into separate vectors and must be 2 or 4.

The quad_shuffle_and_fill_down() function shifts each quad-group downward by delta threads -- similar to quad_shuffle_down() -- and assigns the values from the lower filling lanes to the upper data lanes by wrapping the quad lane IDs. For example, with a delta value of 2, the function:
- Shifts the quad lane IDs up by two
- Assigns the lower two lanes of filling to the upper two lanes of data

Computed
quad lane ID
2
3
4
5
data
c
d
fa
fb

---

### quad_shuffle_and_fill_down (with modulo parameter)

The quad_shuffle_and_fill_down() function with the modulo parameter splits the quad-group into vectors, each with size modulo and shifts each vector by the delta threads. For example, with a modulo value of 2 and a delta value of 1, the function:
- Shifts the quad lane IDs up by one
- Assigns the lower lane of each vector in filling to the upper lane of each vector in data

Computed
quad lane ID
1
2
1
2
data
b
fa
d
fc

---

### quad_ballot

The quad_ballot function uses the quad_vote wrapper type, which can be explicitly cast to its underlying type. (In the following example, note use of vote_t to represent an underlying type, XXX.)

## SIMD-Group Matrix Operations

### simdgroup_matrix<T,Cols,Rows>

2.4 SIMD-group Matrix Data Types

All OS: Metal 2.3 and later support SIMD-group matrix types.

Metal supports a matrix type `simdgroup_matrix<T,Cols,Rows>` defined in `<metal_simdgroup_matrix>`. Operations on SIMD-group matrices are executed cooperatively by threads in the SIMD-group. Therefore, all operations must be executed only under uniform control-flow within the SIMD-group or the behavior is undefined.

Metal supports the following SIMD-group matrix type names, where T is half, bfloat (in Metal 3.1 and later) or float and Cols and Rows are 8:
- simdgroup_half8x8
- simdgroup_bfloat8x8 (Metal 3.1 and later)
- simdgroup_float8x8

The mapping of matrix elements to threads in the SIMD-group is unspecified. For a description of which functions Metal supports on SIMD-group matrices, see section 6.7.

---

### simdgroup_matrix<T,Cols,Rows>(T dval)

```metal
simdgroup_matrix<T,Cols,Rows>(T dval)
```

Creates a diagonal matrix with the given value.

---

### make_filled_simdgroup_matrix(T value)

```metal
simdgroup_matrix<T,Cols,Rows> make_filled_simdgroup_matrix(T value)
```

Initializes a SIMD-group matrix filled with the given value.

---

### simdgroup_load(thread simdgroup_matrix<T,Cols,Rows>& d, const threadgroup T *src, ulong elements_per_row, ulong2 matrix_origin, bool transpose_matrix)

```metal
void simdgroup_load(
  thread simdgroup_matrix<T,Cols,Rows>& d,
  const  threadgroup T *src,
  ulong  elements_per_row = Cols,
  ulong2 matrix_origin = 0,
  bool   transpose_matrix = false)
```

Loads data from threadgroup memory into a SIMD-group matrix. The elements_per_row parameter indicates the number of elements in the source memory layout.

---

### simdgroup_load(thread simdgroup_matrix<T,Cols,Rows>& d, const device T *src, ulong elements_per_row, ulong2 matrix_origin, bool transpose_matrix)

```metal
void simdgroup_load(
  thread simdgroup_matrix<T,Cols,Rows>& d,
  const  device T *src,
  ulong  elements_per_row = Cols,
  ulong2 matrix_origin = 0,
  bool   transpose_matrix = false)
```

Loads data from device memory into a SIMD-group matrix. The elements_per_row parameter indicates the number of elements in the source memory layout.

---

### simdgroup_store(thread simdgroup_matrix<T,Cols,Rows> a, threadgroup T *dst, ulong elements_per_row, ulong2 matrix_origin, bool transpose_matrix)

```metal
void simdgroup_store(
  thread simdgroup_matrix<T,Cols,Rows> a,
  threadgroup T *dst,
  ulong  elements_per_row = Cols,
  ulong2 matrix_origin = 0,
  bool   transpose_matrix = false)
```

Stores data from a SIMD-group matrix into threadgroup memory. The elements_per_row parameter indicates the number of elements in the destination memory layout.

---

### simdgroup_store

```metal
void simdgroup_store(
  thread simdgroup_matrix<T,Cols,Rows> a,
  device T *dst,
  ulong  elements_per_row = Cols,
  ulong2 matrix_origin = 0,
  bool   transpose_matrix = false)
```

Stores data from a SIMD-group matrix into device memory. The elements_per_row parameter indicates the number of elements in the destination memory layout.

---

### simdgroup_multiply

```metal
void simdgroup_multiply(
  thread simdgroup_matrix<T,Cols,Rows>& d,
  thread simdgroup_matrix<T,K,Rows>&    a,
  thread simdgroup_matrix<T,Cols,K>&    b)
```

Returns d = a * b

---

### simdgroup_multiply_accumulate

```metal
void simdgroup_multiply_accumulate(
  thread simdgroup_matrix<T,Cols,Rows>& d,
  thread simdgroup_matrix<T,K,Rows>&    a,
  thread simdgroup_matrix<T,Cols,K>&    b,
  thread simdgroup_matrix<T,Cols,Rows>& c)
```

Returns d = a * b + c

## Integer Math and Bit Manipulation

### clamp

```metal
T clamp(T x, T minval, T maxval)
```

Returns min(max(x, minval), maxval). Results are undefined if minval > maxval.

---

### clz

```metal
T clz(T x)
```

Returns the number of leading 0-bits in x, starting at the most significant bit position. If x is 0, returns the size in bits of the type of x or component type of x, if x is a vector.

---

### ctz

```metal
T ctz(T x)
```

Returns the count of trailing 0-bits in x. If x is 0, returns the size in bits of the type of x or if x is a vector, the component type of x.

---

### extract_bits

```metal
T extract_bits(T x, uint offset, uint bits)
```

**Availability:** All OS: Metal 1.2 and later

Extract bits [offset, offset+bits-1] from x, returning them in the least significant bits of the result.

For unsigned data types, the most significant bits of the result are set to zero. For signed data types, the most significant bits are set to the value of bit offset+bits-1.

If bits is zero, the result is zero. If the sum of offset and bits is greater than the number of bits used to store the operand, the result is undefined.

---

### hadd

```metal
T hadd(T x, T y)
```

Returns (x + y) >> 1. The intermediate sum does not modulo overflow.

---

### insert_bits

```metal
T insert_bits(T base, T insert, uint offset, uint bits)
```

**Availability:** All OS: Metal 1.2 and later

Returns the insertion of the bits least-significant bits of insert into base.

The result has bits [offset, offset+bits-1] taken from bits [0, bits-1] of insert, and all other bits are taken directly from the corresponding bits of base. If bits is zero, the result is base. If the sum of offset and bits is greater than the number of bits used to store the operand, the result is undefined.

---

### mad24

```metal
T32 mad24(T32 x, T32 y, T32 z)
```

**Availability:** All OS: Metal 2.1 and later

Uses mul24 to multiply two 24-bit integer values x and y, adds the 32-bit integer result to the 32-bit integer z, and returns that sum.

---

### madhi

```metal
T madhi(T a, T b, T c)
```

Returns mulhi(a, b) + c.

---

### madsat

```metal
T madsat(T a, T b, T c)
```

Returns a * b + c and saturates the result.

## Indirect Compute Commands

### compute_command

```metal
struct compute_command {
public:
    explicit compute_command(command_buffer icb,
                             unsigned cmd_index);

    void set_compute_pipeline_state(
             compute_pipeline_state pipeline);

    template <typename T …>
    void set_kernel_buffer(device T *buffer, uint index);
    template <typename T …>
    void set_kernel_buffer(constant T *buffer, uint index);

    // Metal 3.1: Supports passing kernel strides.
    template <typename T …>
    void set_kernel_buffer(device T *buffer, size_t stride,
                           uint index);
    template <typename T …>
    void set_kernel_buffer(constant T *buffer, size_t stride,
                           uint index);

    void set_barrier();
    void clear_barrier();

    void concurrent_dispatch_threadgroups(
             uint3 threadgroups_per_grid,
             uint3 threads_per_threadgroup);
    void concurrent_dispatch_threads(uint3 threads_per_grid,
                                     uint3 threads_per_threadgroup);

    void set_threadgroup_memory_length(uint length, uint index);
    void set_stage_in_region(uint3 origin, uint3 size);
};
```


cmake_minimum_required(VERSION 3.20.0)

# Set XNNPACK directory
set(XNNPACK_DIR /scratch/kchern2/zephyr-chipyard-sw/third-party/XNNPACK)

# Set XNNPACK build options (disable tests and benchmarks for faster builds)
set(XNNPACK_ENABLE_RISCV_VECTOR ON CACHE BOOL "Build XNNPACK with RISC-V Vector micro-kernels")
set(XNNPACK_BUILD_TESTS OFF CACHE BOOL "Build XNNPack unit tests")
set(XNNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "Build XNNPack benchmarks")

# Configuration files
set(CONF_FILE prj.conf riscv_vector.conf)

# Find Zephyr
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(rvv_bench)

# Build XNNPACK as a subproject (this builds pthreadpool automatically)
add_subdirectory(${XNNPACK_DIR} xnnpack_build)

# Create interface library to pass compile flags to pthreadpool
add_library(compile_commands INTERFACE)
target_include_directories(compile_commands INTERFACE $<TARGET_PROPERTY:app,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_options(compile_commands INTERFACE $<TARGET_PROPERTY:app,INTERFACE_COMPILE_OPTIONS>)
target_compile_definitions(compile_commands INTERFACE $<TARGET_PROPERTY:app,INTERFACE_COMPILE_DEFINITIONS>)
add_dependencies(compile_commands zephyr_interface)
add_dependencies(compile_commands zephyr_property_target)
target_link_libraries(pthreadpool PUBLIC compile_commands)

# Add XNNPACK include directories
target_include_directories(app PRIVATE ${XNNPACK_DIR})
target_include_directories(app PRIVATE ${XNNPACK_DIR}/include)
target_include_directories(app PRIVATE ${XNNPACK_DIR}/src)

# Link XNNPACK and pthreadpool
target_link_libraries(app PRIVATE XNNPACK pthreadpool)

# Add source files (kernel is embedded in main.c)
target_sources(app PRIVATE src/main.c)

# Enable RVV intrinsics
target_compile_options(app PRIVATE -march=rv64gcv
-O3
-ftree-vectorize
-fopt-info-vec-all)
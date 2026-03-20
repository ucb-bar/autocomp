cmake_minimum_required(VERSION 3.20.0)

# Set XNNPACK directory
set(XNNPACK_DIR $ENV{ZEPHYR_BASE}/../../third-party/XNNPACK)

# Set XNNPACK build options
set(XNNPACK_ENABLE_RISCV_VECTOR ON CACHE BOOL "Build XNNPACK with RISC-V Vector micro-kernels")
set(XNNPACK_BUILD_TESTS OFF CACHE BOOL "Build XNNPack unit tests")
set(XNNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "Build XNNPack benchmarks")

# Configuration files
set(CONF_FILE prj.conf riscv_vector.conf)

# Find Zephyr
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(rvv_bench)

# Create interface library to pass Zephyr's sysroot/includes to subprojects
add_library(compile_commands INTERFACE)
target_include_directories(compile_commands INTERFACE $<TARGET_PROPERTY:app,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_options(compile_commands INTERFACE $<TARGET_PROPERTY:app,INTERFACE_COMPILE_OPTIONS>)
target_compile_definitions(compile_commands INTERFACE $<TARGET_PROPERTY:app,INTERFACE_COMPILE_DEFINITIONS>)
add_dependencies(compile_commands zephyr_interface)
add_dependencies(compile_commands zephyr_property_target)

# Build XNNPACK as a subproject (this builds pthreadpool automatically)
add_subdirectory(${XNNPACK_DIR} xnnpack_build)
target_link_libraries(pthreadpool PUBLIC compile_commands)

# Download and build Google Test for microkernel testing
set(GOOGLETEST_SOURCE_DIR "${CMAKE_BINARY_DIR}/googletest-source")
if(NOT EXISTS "${GOOGLETEST_SOURCE_DIR}/CMakeLists.txt")
  configure_file(
    "${XNNPACK_DIR}/cmake/DownloadGoogleTest.cmake"
    "${CMAKE_BINARY_DIR}/googletest-download/CMakeLists.txt")
  execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download")
  execute_process(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download")
endif()

# Disable POSIX features that don't exist on bare-metal Zephyr
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(gtest_disable_pthreads ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK ON CACHE BOOL "" FORCE)

add_subdirectory(
  "${GOOGLETEST_SOURCE_DIR}"
  "${CMAKE_BINARY_DIR}/googletest"
  EXCLUDE_FROM_ALL)

# GTest: disable POSIX features unavailable on bare-metal Zephyr
# - FILE_SYSTEM=0: skips getcwd/chdir
# - STREAM_REDIRECTION=0: skips dup/dup2/STDOUT_FILENO/STDERR_FILENO
# - POSIX_RE=0: skips POSIX regex
# - PTHREAD=0: no pthreads
# PUBLIC on gtest propagates to gmock (which depends on gtest)
target_compile_definitions(gtest PUBLIC
  GTEST_HAS_FILE_SYSTEM=0
  GTEST_HAS_STREAM_REDIRECTION=0
  GTEST_HAS_POSIX_RE=0
  GTEST_HAS_PTHREAD=0
)

# Force-include stub for isatty() which picolibc doesn't provide.
# PUBLIC so the stub is visible both when compiling GTest itself
# and when any consumer (app, gmock) includes GTest headers.
target_compile_options(gtest PUBLIC
  -include ${CMAKE_CURRENT_SOURCE_DIR}/gtest_zephyr_stubs.h
)

# GTest needs Zephyr's sysroot and include paths for cross-compilation
target_link_libraries(gtest PUBLIC compile_commands)
target_link_libraries(gmock PUBLIC compile_commands)

# Add XNNPACK include directories (root, include, src, and test for tester headers)
target_include_directories(app PRIVATE ${XNNPACK_DIR})
target_include_directories(app PRIVATE ${XNNPACK_DIR}/include)
target_include_directories(app PRIVATE ${XNNPACK_DIR}/src)
target_include_directories(app PRIVATE ${XNNPACK_DIR}/test)

# Link XNNPACK, pthreadpool, and GTest
target_link_libraries(app PRIVATE XNNPACK pthreadpool GTest::gtest GTest::gmock)

# Add source files (kernel is embedded in main.cpp)
target_sources(app PRIVATE src/main.cpp)

# Enable RVV intrinsics
target_compile_options(app PRIVATE -march=rv64gcv
-mabi=lp64d
-O2)

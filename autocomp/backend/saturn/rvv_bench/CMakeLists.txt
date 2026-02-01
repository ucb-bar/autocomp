cmake_minimum_required(VERSION 3.20.0)

# Configuration files
set(CONF_FILE prj.conf riscv_vector.conf)

# Find Zephyr
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(rvv_bench)

# Add source files (kernel is embedded in main.c)
target_sources(app PRIVATE src/main.c)

# Enable RVV intrinsics
target_compile_options(app PRIVATE -march=rv64gcv
-O3
-ftree-vectorize
-fopt-info-vec-all)
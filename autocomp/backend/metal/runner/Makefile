METAL_CPP  := metal-cpp
BUILD_DIR  := build

CXX        := clang++
CXXFLAGS   := -std=c++20 -O2 -Wall -I$(METAL_CPP) -I.
LDFLAGS    := -framework Metal -framework Foundation

METAL      := xcrun metal
METALLIB   := xcrun metallib
METALFLAGS := -std=metal4.0

# ── setup: download Apple metal-cpp headers ─────────────────────
.PHONY: setup
setup:
	@if [ -d "$(METAL_CPP)" ]; then \
		echo "metal-cpp already exists, skipping download."; \
	else \
		echo "Downloading metal-cpp headers from Apple..."; \
		curl -fsSL https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15_iOS18.zip -o /tmp/metal-cpp.zip; \
		unzip -q /tmp/metal-cpp.zip -d .; \
		rm /tmp/metal-cpp.zip; \
		echo "metal-cpp headers installed."; \
	fi

# ── build harness binary ────────────────────────────────────────
# Usage: make harness PROBLEM=0_gemm_q8_f32 HARNESS_DIR=../../../../harnesses/metal-m2
.PHONY: harness
harness: $(BUILD_DIR)/$(PROBLEM)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/$(PROBLEM): $(HARNESS_DIR)/$(PROBLEM).cpp kernel-runner.cpp kernel-runner.hpp harness-utils.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(HARNESS_DIR)/$(PROBLEM).cpp kernel-runner.cpp

# ── compile .metal kernel to .metallib ──────────────────────────
# Usage: make metallib KERNEL_SRC=path/to/kernel.metal
# Output: build/<kernel_basename>.metallib
KERNEL_BASE = $(basename $(notdir $(KERNEL_SRC)))

.PHONY: metallib
metallib: $(BUILD_DIR)/$(KERNEL_BASE).metallib

$(BUILD_DIR)/$(KERNEL_BASE).air: $(KERNEL_SRC) | $(BUILD_DIR)
	$(METAL) $(METALFLAGS) -c $< -o $@

$(BUILD_DIR)/$(KERNEL_BASE).metallib: $(BUILD_DIR)/$(KERNEL_BASE).air
	$(METALLIB) $< -o $@

# ── clean ───────────────────────────────────────────────────────
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)

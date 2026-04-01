#!/bin/bash
# AutoComp optimization runner -- runs beam search on 3 internal kernels
# Usage: bash run_optimize.sh [prob_id]
#   If prob_id is provided, runs only that kernel. Otherwise runs all 3.

set -e

source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
export AWS_REGION=us-east-1
export WANDB_MODE=disabled

cd /home/ubuntu/autocomp

# Install package if needed
pip install -e . -q 2>/dev/null

# Kernels to optimize:
#   5 = FFT256 (177 LOC, no loops, transpose-heavy)
#   6 = Mamba Scan (126 LOC, tensor_tensor_scan)
#   4 = Triangular Multiply (100 LOC, batched matmul)

if [ -n "$1" ]; then
    KERNELS="$1"
else
    KERNELS="5 6 4"
fi

# Warm-up: run each kernel's test harness once to trigger library rehydration
# and first-time NKI compilation. This ensures beam search iterations hit warm caches.
echo "=========================================="
echo "Warm-up phase: pre-compiling kernels"
echo "Start time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=========================================="

for PROB_ID in $KERNELS; do
    TEST_FILE=$(ls tests/trn-internal/${PROB_ID}_*_test.py 2>/dev/null | head -1)
    if [ -n "$TEST_FILE" ]; then
        echo "Warming up: $TEST_FILE"
        python "$TEST_FILE" 2>&1 || echo "Warm-up for prob_id=$PROB_ID returned non-zero (may be OK)"
    else
        echo "WARNING: No test harness found for prob_id=$PROB_ID, skipping warm-up"
    fi
done

echo "Warm-up complete at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

for PROB_ID in $KERNELS; do
    case $PROB_ID in
        5) NAME="fft256" ;;
        6) NAME="mamba_scan" ;;
        4) NAME="trimul" ;;
        *) NAME="kernel_${PROB_ID}" ;;
    esac

    echo "=========================================="
    echo "Optimizing kernel: $NAME (prob_id=$PROB_ID)"
    echo "Start time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "=========================================="

    # Patch run_search.py with the target prob_id
    sed -i "s/prob_id = [0-9]*/prob_id = $PROB_ID/" autocomp/search/run_search.py

    # Run the beam search
    python -m autocomp.search.run_search 2>&1 | tee "output/optimize_${NAME}_$(date -u '+%Y%m%d_%H%M%S').log"

    echo ""
    echo "Completed: $NAME at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo ""
done

echo "All optimizations complete at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

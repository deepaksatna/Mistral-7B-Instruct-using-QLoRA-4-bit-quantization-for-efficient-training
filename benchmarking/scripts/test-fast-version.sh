#!/bin/bash
# Test the FAST version that targets the REAL bottleneck
# Disables gradient checkpointing and uses larger batch size

NODE_ID=${1:-2}
POD_NAME="benchmark-node-$NODE_ID"
BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "======================================================================="
echo "Testing FAST Version - Real Bottleneck Fix"
echo "======================================================================="
echo ""
echo "Changes from original:"
echo "  1. ✓ Gradient checkpointing DISABLED (was slowing by 40%)"
echo "  2. ✓ Larger batch size (16 vs 4)"
echo "  3. ✓ Regular AdamW optimizer (not 8-bit paged)"
echo "  4. ✓ Pre-tokenized dataset (bonus)"
echo ""
echo "Expected: 3-6× faster than original (0.67 → 2.0-4.0 samples/sec)"
echo ""

# Check pod
if ! kubectl get pod $POD_NAME &>/dev/null; then
    echo -e "${RED}✗ ERROR: Pod $POD_NAME not found${NC}"
    exit 1
fi

echo "[1/3] Copying FAST benchmark script..."
kubectl cp "$BENCHMARK_DIR/benchmarks/qlora_benchmark_fast.py" $POD_NAME:/benchmarks/
kubectl exec $POD_NAME -- chmod +x /benchmarks/qlora_benchmark_fast.py
echo -e "${GREEN}✓ Script copied${NC}"
echo ""

echo "[2/3] Cleaning old results..."
kubectl exec $POD_NAME -- rm -rf /results/qlora/node${NODE_ID}_fast_*
echo -e "${GREEN}✓ Clean${NC}"
echo ""

echo "[3/3] Running FAST benchmark (batch 16, no gradient checkpointing)..."
echo ""
echo -e "${YELLOW}This will take ~2-5 minutes (vs 10 minutes original)${NC}"
echo ""

kubectl exec -it $POD_NAME -- python3 /benchmarks/qlora_benchmark_fast.py \
    --node-id $NODE_ID \
    --quick \
    --output-dir /results/qlora

echo ""
echo "======================================================================="
echo "FAST Benchmark Complete!"
echo "======================================================================="
echo ""

# Get results
FAST_FILE="/results/qlora/node${NODE_ID}_fast_4bit_nogc_b16_seq1024_r16.json"

if kubectl exec $POD_NAME -- test -f $FAST_FILE; then
    echo "Comparing with original results..."
    echo ""

    FAST_SAMPLES=$(kubectl exec $POD_NAME -- cat $FAST_FILE | jq -r '.performance.samples_per_second')
    FAST_TIME=$(kubectl exec $POD_NAME -- cat $FAST_FILE | jq -r '.performance.total_train_time_seconds')
    FAST_SPEEDUP=$(kubectl exec $POD_NAME -- cat $FAST_FILE | jq -r '.performance.speedup_vs_baseline')
    FAST_GPU=$(kubectl exec $POD_NAME -- cat $FAST_FILE | jq -r '.performance."estimated_gpu_utilization_%"')

    echo "Results:"
    echo "-----------------------------------------------------------------------"
    printf "%-25s %15s %15s\n" "Metric" "Original" "FAST"
    echo "-----------------------------------------------------------------------"
    printf "%-25s %15.2f %15.2f\n" "Samples/second" 0.67 $FAST_SAMPLES
    printf "%-25s %15.0f %15.0f\n" "Training time (sec)" 596 $FAST_TIME
    printf "%-25s %14.1f%% %14.1f%%\n" "GPU utilization" 22.4 $FAST_GPU
    printf "%-25s %15s %14.1fx\n" "Speedup" "1.0×" $FAST_SPEEDUP
    echo "======================================================================="
    echo ""

    if (( $(echo "$FAST_SPEEDUP > 2.0" | bc -l) )); then
        echo -e "${GREEN}✓ SUCCESS: ${FAST_SPEEDUP}× speedup achieved!${NC}"
    elif (( $(echo "$FAST_SPEEDUP > 1.5" | bc -l) )); then
        echo -e "${YELLOW}⚠ PARTIAL: ${FAST_SPEEDUP}× speedup (expected >2×)${NC}"
    else
        echo -e "${RED}✗ FAILED: Only ${FAST_SPEEDUP}× speedup${NC}"
        echo ""
        echo "Possible issues:"
        echo "  - Gradient checkpointing still enabled (check logs)"
        echo "  - OOM with batch 16 (fell back to smaller batch)"
        echo "  - Other bottleneck (need profiling)"
    fi

    echo ""
    echo "Detailed results:"
    echo "  $FAST_FILE"
    echo ""
    echo "To copy results:"
    echo "  kubectl cp $POD_NAME:$FAST_FILE ./results/fast_result.json"
else
    echo -e "${RED}✗ Results file not found - benchmark may have failed${NC}"
    echo ""
    echo "Check logs for errors"
fi

echo ""

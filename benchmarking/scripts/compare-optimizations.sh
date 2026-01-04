#!/bin/bash
# Compare unoptimized vs optimized QLoRA benchmark
# Run both versions on the same node and compare results

NODE_ID=${1:-2}
POD_NAME="benchmark-node-$NODE_ID"
BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "======================================================================="
echo "Optimization Comparison - Node $NODE_ID"
echo "======================================================================="
echo ""
echo "This will run BOTH versions and compare:"
echo "  1. Original (unoptimized) benchmark"
echo "  2. Optimized benchmark (pre-tokenized + multi-worker)"
echo ""
echo "Expected improvement: 10-15× faster, 80%+ GPU utilization"
echo ""

# Check pod exists
if ! kubectl get pod $POD_NAME &>/dev/null; then
    echo "✗ ERROR: Pod $POD_NAME not found"
    exit 1
fi

echo "[1/5] Copying benchmark scripts to pod..."
kubectl cp "$BENCHMARK_DIR/benchmarks/qlora_benchmark.py" $POD_NAME:/benchmarks/qlora_benchmark.py
kubectl cp "$BENCHMARK_DIR/benchmarks/qlora_benchmark_optimized.py" $POD_NAME:/benchmarks/qlora_benchmark_optimized.py
kubectl exec $POD_NAME -- chmod +x /benchmarks/*.py
echo "✓ Scripts copied"
echo ""

# Clean up old results
echo "[2/5] Cleaning up old results..."
kubectl exec $POD_NAME -- rm -rf /results/qlora/*
echo "✓ Clean"
echo ""

# Run unoptimized benchmark
echo "[3/5] Running UNOPTIMIZED benchmark (batch 4 only)..."
echo "  Expected: ~10 minutes, 22% GPU util, 0.67 samples/sec"
echo ""

kubectl exec $POD_NAME -- python3 /benchmarks/qlora_benchmark.py \
    --node-id $NODE_ID \
    --model-path /models/mistralai--Mistral-7B-Instruct-v0.3 \
    --output-dir /results/qlora \
    --quick

UNOPT_FILE="/results/qlora/node${NODE_ID}_b4_seq1024_r16_ga1.json"

echo ""
echo "[4/5] Running OPTIMIZED benchmark (batch 4 only)..."
echo "  Expected: ~40 seconds, 80%+ GPU util, 10+ samples/sec"
echo ""

kubectl exec $POD_NAME -- python3 /benchmarks/qlora_benchmark_optimized.py \
    --node-id $NODE_ID \
    --model-path /models/mistralai--Mistral-7B-Instruct-v0.3 \
    --output-dir /results/qlora \
    --quick

OPT_FILE="/results/qlora/node${NODE_ID}_opt_b4_seq1024_r16_ga1.json"

echo ""
echo "[5/5] Comparing results..."
echo ""

# Extract metrics and compare
UNOPT_SAMPLES=$(kubectl exec $POD_NAME -- cat $UNOPT_FILE | jq -r '.performance.samples_per_second')
UNOPT_TOKENS=$(kubectl exec $POD_NAME -- cat $UNOPT_FILE | jq -r '.performance.tokens_per_second')
UNOPT_TIME=$(kubectl exec $POD_NAME -- cat $UNOPT_FILE | jq -r '.performance.total_train_time_seconds')
UNOPT_GPU=$(kubectl exec $POD_NAME -- cat $UNOPT_FILE | jq -r '.performance."estimated_gpu_utilization_%"')
UNOPT_COST=$(kubectl exec $POD_NAME -- cat $UNOPT_FILE | jq -r '.cost.cost_per_1000_samples_usd')

OPT_SAMPLES=$(kubectl exec $POD_NAME -- cat $OPT_FILE | jq -r '.performance.samples_per_second')
OPT_TOKENS=$(kubectl exec $POD_NAME -- cat $OPT_FILE | jq -r '.performance.tokens_per_second')
OPT_TIME=$(kubectl exec $POD_NAME -- cat $OPT_FILE | jq -r '.performance.total_train_time_seconds')
OPT_GPU=$(kubectl exec $POD_NAME -- cat $OPT_FILE | jq -r '.performance."estimated_gpu_utilization_%"')
OPT_COST=$(kubectl exec $POD_NAME -- cat $OPT_FILE | jq -r '.cost.cost_per_1000_samples_usd')
OPT_SPEEDUP=$(kubectl exec $POD_NAME -- cat $OPT_FILE | jq -r '.performance.speedup_vs_unoptimized')

echo "======================================================================="
echo "Performance Comparison Results"
echo "======================================================================="
echo ""
printf "%-25s %15s %15s %12s\n" "Metric" "Unoptimized" "Optimized" "Improvement"
echo "-----------------------------------------------------------------------"
printf "%-25s %15.2f %15.2f %11.1fx\n" "Samples/second" $UNOPT_SAMPLES $OPT_SAMPLES $OPT_SPEEDUP
printf "%-25s %15.0f %15.0f %11.1fx\n" "Tokens/second" $UNOPT_TOKENS $OPT_TOKENS $(echo "$OPT_TOKENS / $UNOPT_TOKENS" | bc -l)
printf "%-25s %15.0f %15.0f %11.1fx\n" "Total time (sec)" $UNOPT_TIME $OPT_TIME $(echo "$UNOPT_TIME / $OPT_TIME" | bc -l)
printf "%-25s %14.1f%% %14.1f%% %11.1fx\n" "GPU utilization" $UNOPT_GPU $OPT_GPU $(echo "$OPT_GPU / $UNOPT_GPU" | bc -l)
printf "%-25s %14.4f %14.4f %11.1fx\n" "Cost/1K samples (\$)" $UNOPT_COST $OPT_COST $(echo "$UNOPT_COST / $OPT_COST" | bc -l)
echo "======================================================================="
echo ""

echo "Summary:"
echo "  ✓ Optimized version is ${OPT_SPEEDUP}× faster"
echo "  ✓ GPU utilization improved from ${UNOPT_GPU}% to ${OPT_GPU}%"
echo "  ✓ Cost reduced by $(echo "scale=0; (1 - $OPT_COST / $UNOPT_COST) * 100" | bc)%"
echo ""

echo "Detailed results saved:"
echo "  Unoptimized: $UNOPT_FILE"
echo "  Optimized: $OPT_FILE"
echo ""

echo "To copy results to local machine:"
echo "  kubectl cp $POD_NAME:/results/qlora/ ./results/comparison/"
echo ""

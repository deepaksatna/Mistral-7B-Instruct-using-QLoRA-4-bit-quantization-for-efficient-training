#!/bin/bash
# Run benchmark on a single node (for testing)
# Usage: ./run-single-node.sh [node-number] [--quick]

set -e

# Parse arguments
NODE_ID=${1:-1}
QUICK_MODE=false

if [ "$2" == "--quick" ] || [ "$1" == "--quick" ]; then
    QUICK_MODE=true
fi

POD_NAME="benchmark-node-$NODE_ID"
MODEL_PATH="/models/mistralai--Mistral-7B-Instruct-v0.3"
BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "======================================================================="
echo "Single Node Benchmark Test"
echo "======================================================================="
echo ""
echo "Node ID: $NODE_ID"
echo "Pod: $POD_NAME"
if [ "$QUICK_MODE" = true ]; then
    echo "Mode: Quick (2 configs)"
else
    echo "Mode: Full (12 configs)"
fi
echo ""

# Check pod exists
echo "[1/4] Checking pod status..."
STATUS=$(kubectl get pod $POD_NAME -o jsonpath='{.status.phase}' 2>/dev/null || echo "NotFound")

if [ "$STATUS" != "Running" ]; then
    echo "✗ ERROR: Pod $POD_NAME is not running (status: $STATUS)"
    exit 1
fi

echo "✓ Pod is running"
echo ""

# Copy benchmark script
echo "[2/4] Copying benchmark script..."
kubectl exec $POD_NAME -- mkdir -p /benchmarks
kubectl cp "$BENCHMARK_DIR/benchmarks/qlora_benchmark.py" $POD_NAME:/benchmarks/qlora_benchmark.py
kubectl exec $POD_NAME -- chmod +x /benchmarks/qlora_benchmark.py
echo "✓ Script copied"
echo ""

# Run benchmark
echo "[3/4] Running QLoRA benchmark..."
echo ""

if [ "$QUICK_MODE" = true ]; then
    BENCH_ARGS="--quick"
else
    BENCH_ARGS=""
fi

kubectl exec -it $POD_NAME -- python3 /benchmarks/qlora_benchmark.py \
    --node-id $NODE_ID \
    --model-path $MODEL_PATH \
    --output-dir /results/qlora \
    $BENCH_ARGS

echo ""
echo "[4/4] Results"
echo "======================================================================="
echo ""

# List result files
echo "Result files:"
kubectl exec $POD_NAME -- ls -lh /results/qlora/

echo ""
echo "To view a specific result:"
echo "  kubectl exec $POD_NAME -- cat /results/qlora/node${NODE_ID}_<config>.json"
echo ""
echo "To copy results to local machine:"
echo "  kubectl cp $POD_NAME:/results/qlora/ ./results/qlora-node${NODE_ID}/"
echo ""

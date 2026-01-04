#!/bin/bash
# Run diagnostic benchmark to find the real bottleneck
# Tests 6 different configurations to identify what's limiting performance

NODE_ID=${1:-2}
POD_NAME="benchmark-node-$NODE_ID"
BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "======================================================================="
echo "Diagnostic Benchmark - Finding the Real Bottleneck"
echo "======================================================================="
echo ""
echo "This will test 6 different configurations:"
echo "  1. BASELINE: Current setup (4bit, batch 4, LoRA, GC, 8bit optimizer)"
echo "  2. Regular AdamW optimizer (vs 8-bit paged)"
echo "  3. Smaller LoRA rank (8 vs 16)"
echo "  4. Shorter sequence (512 vs 1024)"
echo "  5. Seq 512 + larger batch (8 vs 4)"
echo "  6. FP16 precision (vs 4-bit)"
echo ""
echo "Expected time: ~15-20 minutes (6 tests × 2-3 min each)"
echo ""

# Check pod
if ! kubectl get pod $POD_NAME &>/dev/null; then
    echo -e "${RED}✗ ERROR: Pod $POD_NAME not found${NC}"
    exit 1
fi

echo "[1/2] Copying diagnostic script..."
kubectl cp "$BENCHMARK_DIR/benchmarks/diagnostic_benchmark.py" $POD_NAME:/benchmarks/
kubectl exec $POD_NAME -- chmod +x /benchmarks/diagnostic_benchmark.py
echo -e "${GREEN}✓ Ready${NC}"
echo ""

echo "[2/2] Running diagnostic tests..."
echo ""
echo -e "${YELLOW}Starting benchmark... this will take ~15-20 minutes${NC}"
echo ""

kubectl exec -it $POD_NAME -- python3 /benchmarks/diagnostic_benchmark.py \
    --node-id $NODE_ID \
    --output-dir /results/qlora

echo ""
echo "======================================================================="
echo "Diagnostic Complete!"
echo "======================================================================="
echo ""

# Get summary
SUMMARY_FILE="/results/qlora/diagnostic_summary_node${NODE_ID}.json"

if kubectl exec $POD_NAME -- test -f $SUMMARY_FILE 2>/dev/null; then
    echo "Copying results to local machine..."
    kubectl cp $POD_NAME:$SUMMARY_FILE ./results/diagnostic_summary.json

    echo ""
    echo "Results saved to: ./results/diagnostic_summary.json"
    echo ""
    echo "To view full summary:"
    echo "  cat ./results/diagnostic_summary.json | jq ."
    echo ""
    echo "To see just the performance comparison:"
    echo "  cat ./results/diagnostic_summary.json | jq '.results[] | {name: .config_name, status: .status, samples_per_sec: .performance.samples_per_second}'"
    echo ""
else
    echo "No summary file found - check for errors in the output above"
fi

echo "======================================================================="

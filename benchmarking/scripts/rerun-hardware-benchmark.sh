#!/bin/bash
# Re-run hardware benchmark after fixing the PyTorch compatibility issue

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "======================================================================"
echo "Re-running Hardware Benchmark (Fixed)"
echo "======================================================================"
echo ""

# Copy fixed script to all nodes
echo "Copying fixed hardware_benchmark.py to all nodes..."
for i in 1 2 3 4; do
    POD_NAME="benchmark-node-$i"
    kubectl cp "$BENCHMARK_DIR/benchmarks/hardware_benchmark.py" $POD_NAME:/benchmarks/
    kubectl exec $POD_NAME -- chmod +x /benchmarks/hardware_benchmark.py
    echo "  ✓ $POD_NAME updated"
done

echo ""
echo -e "${GREEN}✓ Scripts updated${NC}"
echo ""
echo "Running hardware benchmark on all 4 nodes..."
echo ""

# Run in parallel
declare -A PIDS
for i in 1 2 3 4; do
    POD_NAME="benchmark-node-$i"
    kubectl exec $POD_NAME -- python3 /benchmarks/hardware_benchmark.py --node-id $i &
    PIDS[$i]=$!
done

# Wait for completion
FAILED=0
for i in 1 2 3 4; do
    wait ${PIDS[$i]}
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✓ Node $i completed${NC}"
    else
        echo "✗ Node $i failed (exit code $EXIT_CODE)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ Hardware benchmark completed successfully on all nodes${NC}"
else
    echo "⚠ Hardware benchmark failed on $FAILED node(s)"
fi

echo ""
echo "Collecting results..."
mkdir -p "$BENCHMARK_DIR/results/hardware"

for i in 1 2 3 4; do
    kubectl cp benchmark-node-$i:/results/hardware/hardware_node${i}.json \
        "$BENCHMARK_DIR/results/hardware/" 2>/dev/null && \
        echo "  ✓ Node $i results collected" || \
        echo "  ✗ Node $i results not found"
done

echo ""
echo "======================================================================"
echo "Hardware benchmark complete!"
echo "======================================================================"
echo ""
echo "View results:"
echo "  cat results/hardware/hardware_node1.json | jq '.summary'"
echo ""
echo "Now continue with the rest of the suite:"
echo "  ./scripts/run-complete-suite.sh"
echo "  (It will skip hardware and run loading, inference, qlora)"
echo ""

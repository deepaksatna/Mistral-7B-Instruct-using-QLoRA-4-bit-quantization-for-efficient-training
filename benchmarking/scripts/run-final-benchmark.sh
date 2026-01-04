#!/bin/bash
# Run final optimized benchmark on all 4 nodes
# Based on diagnostic findings: seq_length=512, batch_size=8

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "======================================================================"
echo "Final Optimized QLoRA Benchmark - 4 Node Cluster"
echo "======================================================================"
echo ""
echo "Configuration (based on diagnostic results):"
echo "  Batch size: 8 (optimal)"
echo "  Sequence length: 512 (optimal)"
echo "  Expected speedup: 2.11× vs baseline"
echo "  Expected performance: ~1.46 samples/sec per node"
echo ""
echo "This will run 100 training steps on each of 4 nodes in parallel"
echo "Expected time: ~4-5 minutes per node"
echo ""
echo "======================================================================"
echo ""

# Check all pods are running
echo "Checking pod status..."
ALL_READY=true
for i in 1 2 3 4; do
    POD_NAME="benchmark-node-$i"
    if ! kubectl get pod $POD_NAME &>/dev/null; then
        echo -e "${RED}✗ Pod $POD_NAME not found${NC}"
        ALL_READY=false
    else
        STATUS=$(kubectl get pod $POD_NAME -o jsonpath='{.status.phase}')
        if [ "$STATUS" != "Running" ]; then
            echo -e "${YELLOW}⚠ Pod $POD_NAME is $STATUS (not Running)${NC}"
            ALL_READY=false
        else
            echo -e "${GREEN}✓ Pod $POD_NAME is Running${NC}"
        fi
    fi
done

if [ "$ALL_READY" = false ]; then
    echo ""
    echo -e "${RED}ERROR: Not all pods are ready. Please fix pod issues first.${NC}"
    exit 1
fi

echo ""
echo "======================================================================"
echo "[1/3] Ensuring /benchmarks directory exists and copying scripts..."
echo "======================================================================"
echo ""

for i in 1 2 3 4; do
    POD_NAME="benchmark-node-$i"
    echo "Preparing $POD_NAME..."

    # Check if /benchmarks is a file (wrong) and remove it
    if kubectl exec $POD_NAME -- test -f /benchmarks 2>/dev/null; then
        echo "  Removing file /benchmarks (should be directory)..."
        kubectl exec $POD_NAME -- rm -f /benchmarks
    fi

    # Ensure /benchmarks directory exists
    kubectl exec $POD_NAME -- mkdir -p /benchmarks 2>/dev/null || true

    # Copy the script
    echo "  Copying benchmark script..."
    kubectl cp "$BENCHMARK_DIR/benchmarks/qlora_benchmark_final.py" $POD_NAME:/benchmarks/
    kubectl exec $POD_NAME -- chmod +x /benchmarks/qlora_benchmark_final.py
    echo -e "  ${GREEN}✓ Ready${NC}"
done

echo -e "${GREEN}✓ Scripts copied to all nodes${NC}"
echo ""

echo "======================================================================"
echo "[2/3] Running benchmarks on all 4 nodes (in parallel)..."
echo "======================================================================"
echo ""
echo -e "${YELLOW}Starting parallel execution... this will take ~4-5 minutes${NC}"
echo ""

# Run on all nodes in parallel
for i in 1 2 3 4; do
    POD_NAME="benchmark-node-$i"
    echo "Starting benchmark on $POD_NAME..."

    kubectl exec $POD_NAME -- python3 /benchmarks/qlora_benchmark_final.py \
        --node-id $i \
        --output-dir /results/qlora &

    # Store PID for waiting
    PIDS[$i]=$!
done

echo ""
echo "All benchmarks started in parallel. Waiting for completion..."
echo ""

# Wait for all to complete
FAILED=0
for i in 1 2 3 4; do
    wait ${PIDS[$i]}
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✓ Node $i completed successfully${NC}"
    else
        echo -e "${RED}✗ Node $i failed (exit code $EXIT_CODE)${NC}"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "======================================================================"
echo "[3/3] Collecting results from all nodes..."
echo "======================================================================"
echo ""

mkdir -p "$BENCHMARK_DIR/results/final"

for i in 1 2 3 4; do
    POD_NAME="benchmark-node-$i"
    RESULT_FILE="/results/qlora/final_optimized_node${i}.json"

    if kubectl exec $POD_NAME -- test -f $RESULT_FILE 2>/dev/null; then
        echo "Copying results from $POD_NAME..."
        kubectl cp $POD_NAME:$RESULT_FILE "$BENCHMARK_DIR/results/final/node${i}_final.json"
        echo -e "${GREEN}✓ Results saved to results/final/node${i}_final.json${NC}"
    else
        echo -e "${RED}✗ No results file found for $POD_NAME${NC}"
    fi
done

echo ""
echo "======================================================================"
echo "BENCHMARK SUITE COMPLETE"
echo "======================================================================"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All 4 nodes completed successfully!${NC}"
else
    echo -e "${YELLOW}⚠ $FAILED node(s) failed - check logs above${NC}"
fi

echo ""
echo "Results location: $BENCHMARK_DIR/results/final/"
echo ""
echo "To view individual results:"
echo "  cat results/final/node1_final.json | jq ."
echo ""
echo "To compare all nodes:"
echo "  for i in 1 2 3 4; do"
echo "    echo \"Node \$i:\""
echo "    cat results/final/node\${i}_final.json | jq '.performance.samples_per_second'"
echo "  done"
echo ""
echo "To generate summary report:"
echo "  python3 analysis/compare_final_results.py"
echo ""
echo "======================================================================"

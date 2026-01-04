#!/bin/bash
# Run complete benchmark suite on all 4 nodes
# Includes: hardware, loading, inference, and QLoRA training benchmarks

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "======================================================================"
echo "COMPLETE BENCHMARK SUITE - 4 Node A10 Cluster"
echo "======================================================================"
echo ""
echo "This will run the following benchmarks on all 4 nodes:"
echo "  1. Hardware baseline (GPU compute, memory bandwidth)"
echo "  2. Model loading (FP32, FP16, 8-bit, 4-bit)"
echo "  3. Inference (4-bit)"
echo "  4. QLoRA Training (optimized config)"
echo ""
echo "Total estimated time: ~20-25 minutes"
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
echo "[1/5] Copying benchmark scripts to all nodes..."
echo "======================================================================"
echo ""

for i in 1 2 3 4; do
    POD_NAME="benchmark-node-$i"
    echo "Preparing $POD_NAME..."

    # Ensure /benchmarks directory exists
    kubectl exec $POD_NAME -- rm -f /benchmarks 2>/dev/null || true
    kubectl exec $POD_NAME -- mkdir -p /benchmarks 2>/dev/null || true

    # Copy all benchmark scripts
    kubectl cp "$BENCHMARK_DIR/benchmarks/hardware_benchmark.py" $POD_NAME:/benchmarks/ 2>/dev/null || true
    kubectl cp "$BENCHMARK_DIR/benchmarks/loading_benchmark.py" $POD_NAME:/benchmarks/ 2>/dev/null || true
    kubectl cp "$BENCHMARK_DIR/benchmarks/inference_benchmark.py" $POD_NAME:/benchmarks/ 2>/dev/null || true
    kubectl cp "$BENCHMARK_DIR/benchmarks/qlora_benchmark_final.py" $POD_NAME:/benchmarks/ 2>/dev/null || true

    # Make executable
    kubectl exec $POD_NAME -- chmod +x /benchmarks/*.py 2>/dev/null || true

    echo -e "  ${GREEN}✓ Ready${NC}"
done

echo ""
echo -e "${GREEN}✓ All scripts copied${NC}"
echo ""

# Function to run benchmark on all nodes in parallel
run_parallel_benchmark() {
    local BENCHMARK_NAME=$1
    local BENCHMARK_SCRIPT=$2
    local NODE_ARG=$3
    local STEP=$4

    echo "======================================================================"
    echo "[$STEP] Running $BENCHMARK_NAME..."
    echo "======================================================================"
    echo ""
    echo -e "${YELLOW}Starting on all 4 nodes in parallel...${NC}"
    echo ""

    # Start on all nodes
    declare -A PIDS
    for i in 1 2 3 4; do
        POD_NAME="benchmark-node-$i"
        echo "Starting $BENCHMARK_NAME on $POD_NAME..."

        kubectl exec $POD_NAME -- python3 $BENCHMARK_SCRIPT $NODE_ARG $i &
        PIDS[$i]=$!
    done

    # Wait for completion
    echo ""
    echo "Waiting for completion..."
    FAILED=0

    for i in 1 2 3 4; do
        wait ${PIDS[$i]}
        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            echo -e "${GREEN}✓ Node $i completed${NC}"
        else
            echo -e "${RED}✗ Node $i failed (exit code $EXIT_CODE)${NC}"
            FAILED=$((FAILED + 1))
        fi
    done

    echo ""
    if [ $FAILED -eq 0 ]; then
        echo -e "${GREEN}✓ $BENCHMARK_NAME completed successfully on all nodes${NC}"
    else
        echo -e "${YELLOW}⚠ $BENCHMARK_NAME failed on $FAILED node(s)${NC}"
    fi
    echo ""
}

# [2/5] Hardware Baseline Benchmark (~2 min)
run_parallel_benchmark \
    "Hardware Baseline Benchmark" \
    "/benchmarks/hardware_benchmark.py" \
    "--node-id" \
    "2/5"

# [3/5] Model Loading Benchmark (~5 min)
run_parallel_benchmark \
    "Model Loading Benchmark" \
    "/benchmarks/loading_benchmark.py" \
    "--node-id" \
    "3/5"

# [4/5] Inference Benchmark - 4-bit (~3 min)
echo "======================================================================"
echo "[4/5] Running Inference Benchmark (4-bit)..."
echo "======================================================================"
echo ""
echo -e "${YELLOW}Starting on all 4 nodes in parallel...${NC}"
echo ""

declare -A PIDS
for i in 1 2 3 4; do
    POD_NAME="benchmark-node-$i"
    echo "Starting Inference Benchmark on $POD_NAME..."

    kubectl exec $POD_NAME -- python3 /benchmarks/inference_benchmark.py --node-id $i --precision 4bit &
    PIDS[$i]=$!
done

echo ""
echo "Waiting for completion..."
FAILED=0

for i in 1 2 3 4; do
    wait ${PIDS[$i]}
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✓ Node $i completed${NC}"
    else
        echo -e "${RED}✗ Node $i failed${NC}"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ Inference Benchmark completed successfully on all nodes${NC}"
else
    echo -e "${YELLOW}⚠ Inference Benchmark failed on $FAILED node(s)${NC}"
fi
echo ""

# [5/5] QLoRA Training Benchmark (~5 min)
run_parallel_benchmark \
    "QLoRA Training Benchmark (Optimized)" \
    "/benchmarks/qlora_benchmark_final.py" \
    "--node-id" \
    "5/5"

echo "======================================================================"
echo "Collecting results from all nodes..."
echo "======================================================================"
echo ""

mkdir -p "$BENCHMARK_DIR/results/"{hardware,loading,inference,final}

for i in 1 2 3 4; do
    POD_NAME="benchmark-node-$i"
    echo "Collecting from $POD_NAME..."

    # Hardware
    if kubectl cp $POD_NAME:/results/hardware/hardware_node${i}.json \
        "$BENCHMARK_DIR/results/hardware/" 2>/dev/null; then
        echo -e "  ${GREEN}✓ Hardware results${NC}"
    else
        echo -e "  ${YELLOW}⚠ Hardware results not found${NC}"
    fi

    # Loading
    if kubectl cp $POD_NAME:/results/loading/loading_node${i}.json \
        "$BENCHMARK_DIR/results/loading/" 2>/dev/null; then
        echo -e "  ${GREEN}✓ Loading results${NC}"
    else
        echo -e "  ${YELLOW}⚠ Loading results not found${NC}"
    fi

    # Inference
    if kubectl cp $POD_NAME:/results/inference/inference_4bit_node${i}.json \
        "$BENCHMARK_DIR/results/inference/" 2>/dev/null; then
        echo -e "  ${GREEN}✓ Inference results${NC}"
    else
        echo -e "  ${YELLOW}⚠ Inference results not found${NC}"
    fi

    # QLoRA
    if kubectl cp $POD_NAME:/results/qlora/final_optimized_node${i}.json \
        "$BENCHMARK_DIR/results/final/" 2>/dev/null; then
        echo -e "  ${GREEN}✓ QLoRA results${NC}"
    else
        echo -e "  ${YELLOW}⚠ QLoRA results not found${NC}"
    fi

    echo ""
done

echo "======================================================================"
echo "BENCHMARK SUITE COMPLETE!"
echo "======================================================================"
echo ""
echo "Results saved to:"
echo "  $BENCHMARK_DIR/results/hardware/"
echo "  $BENCHMARK_DIR/results/loading/"
echo "  $BENCHMARK_DIR/results/inference/"
echo "  $BENCHMARK_DIR/results/final/"
echo ""
echo "To view comprehensive analysis:"
echo "  python3 analysis/generate_full_report.py"
echo ""
echo "To view individual benchmark results:"
echo "  # QLoRA training comparison"
echo "  python3 analysis/compare_final_results.py"
echo ""
echo "  # Hardware results"
echo "  cat results/hardware/hardware_node1.json | jq '.summary'"
echo ""
echo "  # Loading results"
echo "  cat results/loading/loading_node1.json | jq '.precision_results'"
echo ""
echo "  # Inference results"
echo "  cat results/inference/inference_4bit_node1.json | jq '.latency'"
echo ""
echo "======================================================================"

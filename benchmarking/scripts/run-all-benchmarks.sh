#!/bin/bash
# Master script to run comprehensive benchmarks across all 4 A10 GPU nodes
# Executes benchmarks in parallel and monitors progress

set -e

# Configuration
BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PODS=("benchmark-node-1" "benchmark-node-2" "benchmark-node-3" "benchmark-node-4")
MODEL_PATH="/models/mistralai--Mistral-7B-Instruct-v0.3"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "======================================================================="
echo "LLM Training Benchmark Suite - 4× A10 GPU Nodes"
echo "======================================================================="
echo ""

# Parse arguments
QUICK_MODE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--quick] [--dry-run]"
            exit 1
            ;;
    esac
done

if [ "$QUICK_MODE" = true ]; then
    echo -e "${YELLOW}⚡ Quick mode enabled (2 configs per benchmark)${NC}"
fi

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}🔍 Dry run mode (no actual execution)${NC}"
fi

echo ""

# Step 1: Verify all pods are running
echo "[1/6] Verifying benchmark pods..."
echo ""

POD_COUNT=0
for pod in "${PODS[@]}"; do
    STATUS=$(kubectl get pod $pod -o jsonpath='{.status.phase}' 2>/dev/null || echo "NotFound")
    if [ "$STATUS" == "Running" ]; then
        echo -e "  ${GREEN}✓${NC} $pod: Running"
        ((POD_COUNT++))
    else
        echo -e "  ${RED}✗${NC} $pod: $STATUS"
    fi
done

echo ""

if [ $POD_COUNT -ne 4 ]; then
    echo -e "${RED}✗ ERROR: Expected 4 running pods, found $POD_COUNT${NC}"
    echo ""
    echo "Please deploy benchmark pods first:"
    echo "  kubectl apply -f manifests/benchmark-pods.yaml"
    exit 1
fi

echo -e "${GREEN}✓ All 4 benchmark pods are running${NC}"
echo ""

# Step 2: Copy benchmark scripts to pods
echo "[2/6] Copying benchmark scripts to pods..."
echo ""

for pod in "${PODS[@]}"; do
    echo "  Copying to $pod..."

    # Create benchmarks directory in pod
    kubectl exec $pod -- mkdir -p /benchmarks

    # Copy QLoRA benchmark
    kubectl cp "$BENCHMARK_DIR/benchmarks/qlora_benchmark.py" $pod:/benchmarks/qlora_benchmark.py

    # Make executable
    kubectl exec $pod -- chmod +x /benchmarks/qlora_benchmark.py
done

echo -e "${GREEN}✓ Scripts copied to all pods${NC}"
echo ""

# Step 3: Run QLoRA benchmarks in parallel
echo "[3/6] Launching QLoRA benchmarks on all nodes..."
echo ""

if [ "$QUICK_MODE" = true ]; then
    BENCH_ARGS="--quick"
else
    BENCH_ARGS=""
fi

if [ "$DRY_RUN" = true ]; then
    echo "Would execute on each pod:"
    echo "  python3 /benchmarks/qlora_benchmark.py --node-id {1-4} --model-path $MODEL_PATH --output-dir /results/qlora $BENCH_ARGS"
    echo ""
    exit 0
fi

# Launch benchmarks in background
for i in {1..4}; do
    pod="benchmark-node-$i"
    echo -e "  ${BLUE}▶${NC} Starting benchmark on $pod (node $i)..."

    kubectl exec $pod -- bash -c "
        nohup python3 /benchmarks/qlora_benchmark.py \
            --node-id $i \
            --model-path $MODEL_PATH \
            --output-dir /results/qlora \
            $BENCH_ARGS \
            > /results/qlora_node${i}.log 2>&1 &
        echo \$! > /tmp/benchmark_pid_qlora
    " &
done

echo ""
echo -e "${GREEN}✓ QLoRA benchmarks launched on all 4 nodes${NC}"
echo ""

# Step 4: Monitor progress
echo "[4/6] Monitoring benchmark progress..."
echo ""
echo "Benchmarks are running in background on all pods."
echo "This will take approximately:"
if [ "$QUICK_MODE" = true ]; then
    echo "  - Quick mode: ~15-20 minutes"
else
    echo "  - Full mode: ~60-90 minutes"
fi
echo ""

# Function to check if benchmark is still running
check_running() {
    local pod=$1
    local pid_file=$2

    kubectl exec $pod -- bash -c "
        if [ -f $pid_file ]; then
            pid=\$(cat $pid_file)
            if ps -p \$pid > /dev/null 2>&1; then
                echo 'running'
            else
                echo 'finished'
            fi
        else
            echo 'not_started'
        fi
    " 2>/dev/null
}

# Monitor loop
echo "Monitoring progress (press Ctrl+C to stop monitoring, benchmarks will continue):"
echo ""

COMPLETED=0
while [ $COMPLETED -lt 4 ]; do
    COMPLETED=0

    for i in {1..4}; do
        pod="benchmark-node-$i"
        status=$(check_running $pod "/tmp/benchmark_pid_qlora")

        if [ "$status" == "finished" ]; then
            echo -e "  ${GREEN}✓${NC} Node $i: Completed"
            ((COMPLETED++))
        elif [ "$status" == "running" ]; then
            echo -e "  ${BLUE}▶${NC} Node $i: Running..."
        else
            echo -e "  ${YELLOW}⏳${NC} Node $i: Not started"
        fi
    done

    echo ""

    if [ $COMPLETED -lt 4 ]; then
        echo "Progress: $COMPLETED/4 nodes completed. Checking again in 60 seconds..."
        echo ""
        sleep 60
    fi
done

echo -e "${GREEN}✓ All benchmarks completed!${NC}"
echo ""

# Step 5: Show results summary
echo "[5/6] Results summary..."
echo ""

for i in {1..4}; do
    pod="benchmark-node-$i"
    echo "Node $i ($pod):"

    # Count result files
    RESULT_COUNT=$(kubectl exec $pod -- sh -c "ls /results/qlora/*.json 2>/dev/null | wc -l" || echo "0")
    echo "  Results files: $RESULT_COUNT"

    # Show log tail
    echo "  Last log lines:"
    kubectl exec $pod -- tail -n 3 /results/qlora_node${i}.log 2>/dev/null | sed 's/^/    /' || echo "    (no log available)"
    echo ""
done

# Step 6: Next steps
echo "[6/6] Next steps"
echo "======================================================================="
echo ""
echo "Benchmarks completed! To collect results:"
echo "  ./scripts/collect-results.sh"
echo ""
echo "To analyze results:"
echo "  cd analysis"
echo "  python3 analyze_results.py"
echo "  python3 generate_report.py"
echo ""
echo "To view logs from a specific node:"
echo "  kubectl exec benchmark-node-1 -- cat /results/qlora_node1.log"
echo ""
echo "======================================================================="

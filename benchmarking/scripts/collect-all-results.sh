#!/bin/bash
# Collect all benchmark results from pods

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "======================================================================"
echo "Collecting Results from All Nodes"
echo "======================================================================"
echo ""

# Create local results directories
mkdir -p "$BENCHMARK_DIR/results/"{hardware,loading,inference,final}

for i in 1 2 3 4; do
    POD_NAME="benchmark-node-$i"
    echo "Collecting from $POD_NAME..."

    # Check what files exist in the pod
    echo "  Checking available results..."

    # Hardware results
    if kubectl exec $POD_NAME -- test -f /results/hardware/hardware_node${i}.json 2>/dev/null; then
        kubectl cp $POD_NAME:/results/hardware/hardware_node${i}.json \
            "$BENCHMARK_DIR/results/hardware/hardware_node${i}.json" 2>/dev/null && \
            echo -e "  ${GREEN}✓ Hardware results${NC}" || \
            echo -e "  ${YELLOW}⚠ Hardware copy failed${NC}"
    else
        echo -e "  ${YELLOW}⚠ Hardware results not found${NC}"
    fi

    # Loading results
    if kubectl exec $POD_NAME -- test -f /results/loading/loading_node${i}.json 2>/dev/null; then
        kubectl cp $POD_NAME:/results/loading/loading_node${i}.json \
            "$BENCHMARK_DIR/results/loading/loading_node${i}.json" 2>/dev/null && \
            echo -e "  ${GREEN}✓ Loading results${NC}" || \
            echo -e "  ${YELLOW}⚠ Loading copy failed${NC}"
    else
        echo -e "  ${YELLOW}⚠ Loading results not found${NC}"
    fi

    # Inference results
    if kubectl exec $POD_NAME -- test -f /results/inference/inference_4bit_node${i}.json 2>/dev/null; then
        kubectl cp $POD_NAME:/results/inference/inference_4bit_node${i}.json \
            "$BENCHMARK_DIR/results/inference/inference_4bit_node${i}.json" 2>/dev/null && \
            echo -e "  ${GREEN}✓ Inference results${NC}" || \
            echo -e "  ${YELLOW}⚠ Inference copy failed${NC}"
    else
        echo -e "  ${YELLOW}⚠ Inference results not found${NC}"
    fi

    # QLoRA results (try both possible locations)
    if kubectl exec $POD_NAME -- test -f /results/qlora/final_optimized_node${i}.json 2>/dev/null; then
        kubectl cp $POD_NAME:/results/qlora/final_optimized_node${i}.json \
            "$BENCHMARK_DIR/results/final/node${i}_final.json" 2>/dev/null && \
            echo -e "  ${GREEN}✓ QLoRA results${NC}" || \
            echo -e "  ${YELLOW}⚠ QLoRA copy failed${NC}"
    else
        echo -e "  ${YELLOW}⚠ QLoRA results not found${NC}"
    fi

    echo ""
done

echo "======================================================================"
echo "Results Collection Summary"
echo "======================================================================"
echo ""

# Count collected files
HARDWARE_COUNT=$(ls "$BENCHMARK_DIR/results/hardware/"*.json 2>/dev/null | wc -l)
LOADING_COUNT=$(ls "$BENCHMARK_DIR/results/loading/"*.json 2>/dev/null | wc -l)
INFERENCE_COUNT=$(ls "$BENCHMARK_DIR/results/inference/"*.json 2>/dev/null | wc -l)
QLORA_COUNT=$(ls "$BENCHMARK_DIR/results/final/"*.json 2>/dev/null | wc -l)

echo "Collected results:"
echo "  Hardware:  $HARDWARE_COUNT / 4 nodes"
echo "  Loading:   $LOADING_COUNT / 4 nodes"
echo "  Inference: $INFERENCE_COUNT / 4 nodes"
echo "  QLoRA:     $QLORA_COUNT / 4 nodes"
echo ""

TOTAL_EXPECTED=16
TOTAL_COLLECTED=$((HARDWARE_COUNT + LOADING_COUNT + INFERENCE_COUNT + QLORA_COUNT))

if [ $TOTAL_COLLECTED -eq $TOTAL_EXPECTED ]; then
    echo -e "${GREEN}✓ All results collected successfully!${NC}"
elif [ $TOTAL_COLLECTED -gt 0 ]; then
    echo -e "${YELLOW}⚠ Partial results collected: $TOTAL_COLLECTED / $TOTAL_EXPECTED${NC}"
else
    echo -e "${RED}✗ No results found${NC}"
fi

echo ""
echo "To view comprehensive report:"
echo "  python3 analysis/generate_full_report.py"
echo ""

#!/bin/bash
# Check what result files exist in each pod

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "======================================================================"
echo "Checking Results in All Pods"
echo "======================================================================"
echo ""

for i in 1 2 3 4; do
    POD_NAME="benchmark-node-$i"
    echo "=== $POD_NAME ==="
    echo ""

    echo "Results directory structure:"
    kubectl exec $POD_NAME -- ls -la /results/ 2>/dev/null || echo "  /results/ not found"
    echo ""

    echo "Hardware results:"
    kubectl exec $POD_NAME -- ls -lh /results/hardware/ 2>/dev/null || echo "  Directory not found"
    echo ""

    echo "Loading results:"
    kubectl exec $POD_NAME -- ls -lh /results/loading/ 2>/dev/null || echo "  Directory not found"
    echo ""

    echo "Inference results:"
    kubectl exec $POD_NAME -- ls -lh /results/inference/ 2>/dev/null || echo "  Directory not found"
    echo ""

    echo "QLoRA results:"
    kubectl exec $POD_NAME -- ls -lh /results/qlora/ 2>/dev/null || echo "  Directory not found"
    echo ""

    echo "Final results:"
    kubectl exec $POD_NAME -- ls -lh /results/final/ 2>/dev/null || echo "  Directory not found"
    echo ""

    echo "───────────────────────────────────────────────────────────────────────"
    echo ""
done

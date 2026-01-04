#!/bin/bash
# Fix /benchmarks directory on nodes 3 and 4
# Issue: /benchmarks exists as a file instead of a directory

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "======================================================================"
echo "Fix /benchmarks Directory Structure"
echo "======================================================================"
echo ""

# Check and fix all nodes
for NODE_ID in 1 2 3 4; do
    POD_NAME="benchmark-node-$NODE_ID"

    echo "Checking $POD_NAME..."

    # Check if /benchmarks is a file
    if kubectl exec $POD_NAME -- test -f /benchmarks 2>/dev/null; then
        echo -e "${YELLOW}✗ /benchmarks is a FILE on $POD_NAME (should be directory)${NC}"
        echo "  Fixing..."

        # Remove the file and create directory
        kubectl exec $POD_NAME -- rm -f /benchmarks
        kubectl exec $POD_NAME -- mkdir -p /benchmarks

        echo -e "${GREEN}✓ Fixed: Removed file and created directory${NC}"

    elif kubectl exec $POD_NAME -- test -d /benchmarks 2>/dev/null; then
        echo -e "${GREEN}✓ /benchmarks is already a directory on $POD_NAME${NC}"
    else
        echo "  /benchmarks does not exist, creating..."
        kubectl exec $POD_NAME -- mkdir -p /benchmarks
        echo -e "${GREEN}✓ Created /benchmarks directory${NC}"
    fi

    echo ""
done

echo "======================================================================"
echo "Directory Fix Complete"
echo "======================================================================"
echo ""
echo "Now re-run the benchmark:"
echo "  ./scripts/run-final-benchmark.sh"
echo ""

#!/bin/bash
# Collect benchmark results from all 4 pods to local machine
# Organizes results by benchmark type and node

set -e

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$BENCHMARK_DIR/results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "======================================================================="
echo "Collecting Benchmark Results from All Nodes"
echo "======================================================================="
echo ""
echo "Results will be saved to: $RESULTS_DIR"
echo "Timestamp: $TIMESTAMP"
echo ""

# Pods to collect from
PODS=("benchmark-node-1" "benchmark-node-2" "benchmark-node-3" "benchmark-node-4")

# Step 1: Create local results directories
echo "[1/4] Preparing local directories..."
mkdir -p "$RESULTS_DIR/qlora"
mkdir -p "$RESULTS_DIR/logs"
mkdir -p "$RESULTS_DIR/archive/$TIMESTAMP"
echo "✓ Directories ready"
echo ""

# Step 2: Collect results from each pod
echo "[2/4] Collecting results from pods..."
echo ""

TOTAL_FILES=0

for i in {1..4}; do
    pod="benchmark-node-$i"

    echo -e "${BLUE}Node $i ($pod):${NC}"

    # Check if pod exists
    if ! kubectl get pod $pod &>/dev/null; then
        echo -e "  ${YELLOW}⚠${NC} Pod not found, skipping"
        echo ""
        continue
    fi

    # Check if results directory exists in pod
    if ! kubectl exec $pod -- test -d /results/qlora 2>/dev/null; then
        echo -e "  ${YELLOW}⚠${NC} No results directory found"
        echo ""
        continue
    fi

    # Count files
    FILE_COUNT=$(kubectl exec $pod -- sh -c "ls /results/qlora/*.json 2>/dev/null | wc -l" || echo "0")
    echo "  Found $FILE_COUNT result files"

    if [ "$FILE_COUNT" -gt 0 ]; then
        # Copy QLoRA results
        echo "  Copying QLoRA results..."
        kubectl cp $pod:/results/qlora/ "$RESULTS_DIR/qlora/node$i/" 2>/dev/null || true

        # Copy logs
        echo "  Copying logs..."
        kubectl cp $pod:/results/qlora_node${i}.log "$RESULTS_DIR/logs/node${i}_qlora.log" 2>/dev/null || true

        TOTAL_FILES=$((TOTAL_FILES + FILE_COUNT))
        echo -e "  ${GREEN}✓${NC} Copied successfully"
    else
        echo -e "  ${YELLOW}⚠${NC} No results to copy"
    fi

    echo ""
done

echo -e "${GREEN}✓ Collected $TOTAL_FILES result files from all nodes${NC}"
echo ""

# Step 3: Organize and validate results
echo "[3/4] Organizing results..."
echo ""

# Create summary directory
mkdir -p "$RESULTS_DIR/summary"

# Count results by type
QLORA_COUNT=$(find "$RESULTS_DIR/qlora" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')

echo "Results summary:"
echo "  QLoRA benchmarks: $QLORA_COUNT files"
echo ""

# List all result files
echo "Detailed file list:"
for i in {1..4}; do
    NODE_DIR="$RESULTS_DIR/qlora/node$i"
    if [ -d "$NODE_DIR" ]; then
        FILE_COUNT=$(ls -1 "$NODE_DIR"/*.json 2>/dev/null | wc -l | tr -d ' ')
        echo "  Node $i: $FILE_COUNT files"

        if [ "$FILE_COUNT" -gt 0 ]; then
            ls -1 "$NODE_DIR"/*.json | head -n 3 | sed 's/^/    /'
            if [ "$FILE_COUNT" -gt 3 ]; then
                echo "    ... and $((FILE_COUNT - 3)) more"
            fi
        fi
    fi
done

echo ""

# Step 4: Archive results
echo "[4/4] Creating archive..."

# Copy to archive
cp -r "$RESULTS_DIR/qlora" "$RESULTS_DIR/archive/$TIMESTAMP/" 2>/dev/null || true
cp -r "$RESULTS_DIR/logs" "$RESULTS_DIR/archive/$TIMESTAMP/" 2>/dev/null || true

# Create tar archive
ARCHIVE_FILE="$RESULTS_DIR/archive/benchmark_results_${TIMESTAMP}.tar.gz"
tar -czf "$ARCHIVE_FILE" -C "$RESULTS_DIR" qlora logs 2>/dev/null || true

if [ -f "$ARCHIVE_FILE" ]; then
    ARCHIVE_SIZE=$(du -h "$ARCHIVE_FILE" | cut -f1)
    echo "✓ Archive created: $ARCHIVE_FILE ($ARCHIVE_SIZE)"
else
    echo "⚠ Archive creation skipped (no data)"
fi

echo ""

# Summary
echo "======================================================================="
echo "Results Collection Complete"
echo "======================================================================="
echo ""
echo "Total files collected: $TOTAL_FILES"
echo ""
echo "Results location:"
echo "  QLoRA: $RESULTS_DIR/qlora/"
echo "  Logs: $RESULTS_DIR/logs/"
echo "  Archive: $ARCHIVE_FILE"
echo ""
echo "Next steps:"
echo ""
echo "1. Analyze results:"
echo "   cd $BENCHMARK_DIR/analysis"
echo "   python3 analyze_results.py"
echo ""
echo "2. Compare nodes:"
echo "   python3 compare_nodes.py"
echo ""
echo "3. Generate report:"
echo "   python3 generate_report.py"
echo ""
echo "4. View individual results:"
echo "   cat $RESULTS_DIR/qlora/node1/node1_<config>.json | jq ."
echo ""
echo "5. View logs:"
echo "   cat $RESULTS_DIR/logs/node1_qlora.log"
echo ""
echo "======================================================================="

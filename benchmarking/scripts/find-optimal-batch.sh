#!/bin/bash
# Find optimal batch size without gradient checkpointing
# Tests batch sizes progressively until OOM, then uses the largest that works

NODE_ID=${1:-1}
POD_NAME="benchmark-node-$NODE_ID"
BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "======================================================================="
echo "Finding Optimal Batch Size (No Gradient Checkpointing)"
echo "======================================================================="
echo ""
echo "Strategy: Test batch sizes 4, 6, 8, 10, 12 until OOM"
echo "Expected: Batch 8-10 will work and be 2-3× faster"
echo ""

# Check pod
if ! kubectl get pod $POD_NAME &>/dev/null; then
    echo -e "${RED}✗ ERROR: Pod $POD_NAME not found${NC}"
    exit 1
fi

echo "[1/2] Copying test script..."
kubectl cp "$BENCHMARK_DIR/benchmarks/test_single_batch.py" $POD_NAME:/benchmarks/
kubectl exec $POD_NAME -- chmod +x /benchmarks/test_single_batch.py
echo -e "${GREEN}✓ Ready${NC}"
echo ""

echo "[2/2] Testing batch sizes..."
echo ""

# Test batch sizes
BATCH_SIZES=(4 6 8 10 12)
SUCCESSFUL_BATCH=0
SUCCESSFUL_SAMPLES=0
SUCCESSFUL_TIME=0

for BATCH in "${BATCH_SIZES[@]}"; do
    echo "───────────────────────────────────────────────────────────────────────"
    echo -e "${BLUE}Testing Batch Size: $BATCH${NC}"
    echo "───────────────────────────────────────────────────────────────────────"

    # Run single batch test
    kubectl exec $POD_NAME -- python3 /benchmarks/test_single_batch.py \
        --batch-size $BATCH \
        --node-id $NODE_ID \
        --output-dir /results/qlora

    EXIT_CODE=$?

    RESULT_FILE="/results/qlora/batch_test_node${NODE_ID}_b${BATCH}.json"

    # Check result based on exit code
    if [ $EXIT_CODE -eq 0 ]; then
        # Success - batch fits
        SAMPLES=$(kubectl exec $POD_NAME -- cat $RESULT_FILE | jq -r '.performance.samples_per_second')
        TIME=$(kubectl exec $POD_NAME -- cat $RESULT_FILE | jq -r '.performance.seconds_per_step')
        MEMORY=$(kubectl exec $POD_NAME -- cat $RESULT_FILE | jq -r '.memory.peak_during_training.max_allocated_gb')

        echo -e "${GREEN}✓ SUCCESS: Batch $BATCH works!${NC}"
        echo "  Samples/sec: $SAMPLES"
        echo "  Sec/step: ${TIME}s"
        echo "  Peak memory: ${MEMORY} GB"
        echo ""

        SUCCESSFUL_BATCH=$BATCH
        SUCCESSFUL_SAMPLES=$SAMPLES
        SUCCESSFUL_TIME=$TIME
        SUCCESSFUL_MEMORY=$MEMORY

    elif [ $EXIT_CODE -eq 1 ]; then
        # OOM - stop testing
        echo -e "${YELLOW}✗ OOM: Batch $BATCH too large${NC}"
        echo "  Stopping tests (larger batches will also OOM)"
        echo ""
        break

    else
        # Other error
        echo -e "${RED}✗ ERROR: Batch $BATCH failed${NC}"
        ERROR=$(kubectl exec $POD_NAME -- cat $RESULT_FILE | jq -r '.error // "Unknown error"')
        echo "  Error: $ERROR"
        echo ""
        break
    fi

    # Small delay and cleanup between tests
    kubectl exec $POD_NAME -- python3 -c "import torch, gc; gc.collect(); torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 3
done

echo "======================================================================="
echo "Optimal Batch Size Found"
echo "======================================================================="
echo ""

if [ $SUCCESSFUL_BATCH -gt 0 ]; then
    echo -e "${GREEN}✓ Optimal Batch Size: $SUCCESSFUL_BATCH${NC}"
    echo ""
    echo "Performance:"
    echo "  Samples/sec: $SUCCESSFUL_SAMPLES (vs 0.67 baseline)"
    echo "  Sec/step: ${SUCCESSFUL_TIME}s (vs 6.0s baseline)"
    echo "  Peak memory: ${SUCCESSFUL_MEMORY} GB / 23.68 GB available"
    echo ""

    # Calculate improvement
    IMPROVEMENT=$(echo "scale=1; $SUCCESSFUL_SAMPLES / 0.67" | bc)

    echo "Speedup: ${IMPROVEMENT}× faster than baseline"
    echo ""

    if (( $(echo "$IMPROVEMENT >= 3.0" | bc -l) )); then
        echo -e "${GREEN}✓ EXCELLENT: ${IMPROVEMENT}× improvement!${NC}"
        echo "  This is a major speedup from disabling gradient checkpointing."
    elif (( $(echo "$IMPROVEMENT >= 2.0" | bc -l) )); then
        echo -e "${GREEN}✓ GOOD: ${IMPROVEMENT}× improvement${NC}"
        echo "  Significant speedup achieved."
    elif (( $(echo "$IMPROVEMENT >= 1.5" | bc -l) )); then
        echo -e "${YELLOW}⚠ MODERATE: ${IMPROVEMENT}× improvement${NC}"
        echo "  Some improvement, but could be better."
    else
        echo -e "${RED}✗ MINIMAL: Only ${IMPROVEMENT}× improvement${NC}"
        echo "  May still have other bottlenecks."
    fi

    echo ""
    echo "───────────────────────────────────────────────────────────────────────"
    echo "Recommendation:"
    echo "───────────────────────────────────────────────────────────────────────"
    echo ""
    echo "Use these settings for optimal performance:"
    echo ""
    echo "  TrainingArguments("
    echo "      per_device_train_batch_size=$SUCCESSFUL_BATCH,"
    echo "      gradient_accumulation_steps=1,"
    echo "      optim='adamw_torch',"
    echo "      bf16=True,"
    echo "      dataloader_num_workers=4,"
    echo "  )"
    echo ""
    echo "  # IMPORTANT: Do NOT enable gradient checkpointing"
    echo "  # model.gradient_checkpointing_enable()  # DON'T call this!"
    echo ""
    echo "───────────────────────────────────────────────────────────────────────"
    echo ""
    echo "Results saved to:"
    echo "  /results/qlora/batch_test_node${NODE_ID}_b${SUCCESSFUL_BATCH}.json"
    echo ""
else
    echo -e "${RED}✗ No successful batch size found!${NC}"
    echo ""
    echo "This suggests even batch 4 without gradient checkpointing causes OOM."
    echo ""
    echo "Recommendation:"
    echo "  Use batch 4 WITH gradient checkpointing (original setup)"
    echo "  OR use smaller sequence length (512 instead of 1024)"
    echo ""
    echo "  TrainingArguments("
    echo "      per_device_train_batch_size=4,"
    echo "      gradient_accumulation_steps=4,  # Effective batch = 16"
    echo "  )"
    echo ""
    echo "  model.gradient_checkpointing_enable()  # Needed for memory"
    echo ""
fi

echo "======================================================================="

#!/bin/bash
# Run distributed training benchmark across multiple nodes

set -e

# Configuration
WORLD_SIZE=${1:-4}  # Number of nodes (default: 4)
BATCH_SIZE=${2:-8}  # Batch size per device (default: 8)
SEQ_LENGTH=${3:-512}  # Sequence length (default: 512)
MAX_STEPS=${4:-100}  # Training steps (default: 100)

echo "============================================"
echo "Distributed Training Benchmark"
echo "============================================"
echo "World size: $WORLD_SIZE nodes"
echo "Batch size per device: $BATCH_SIZE"
echo "Global batch size: $((BATCH_SIZE * WORLD_SIZE))"
echo "Sequence length: $SEQ_LENGTH"
echo "Training steps: $MAX_STEPS"
echo "============================================"
echo ""

# Benchmark pods should already be running
# Check if pods exist
echo "Checking benchmark pods..."
for i in $(seq 1 $WORLD_SIZE); do
    if ! kubectl get pod benchmark-node-$i >/dev/null 2>&1; then
        echo "ERROR: benchmark-node-$i not found!"
        echo "Please deploy benchmark pods first:"
        echo "  kubectl apply -f deployment/benchmark-pods.yaml"
        exit 1
    fi
done
echo "✓ All $WORLD_SIZE benchmark pods found"
echo ""

# Copy benchmark script to all pods
echo "Copying benchmark script to all pods..."
for i in $(seq 1 $WORLD_SIZE); do
    kubectl cp distributed_training_benchmark.py benchmark-node-$i:/workspace/benchmark.py
    echo "  ✓ Copied to benchmark-node-$i"
done
echo ""

# Master node is benchmark-node-1
MASTER_POD="benchmark-node-1"
MASTER_ADDR=$(kubectl get pod $MASTER_POD -o jsonpath='{.status.podIP}')
MASTER_PORT=29500

echo "Master node: $MASTER_POD"
echo "Master address: $MASTER_ADDR"
echo ""

# Function to run benchmark on a single node
run_on_node() {
    local NODE_ID=$1
    local RANK=$((NODE_ID - 1))
    local POD_NAME="benchmark-node-$NODE_ID"

    echo "Starting benchmark on node $NODE_ID (rank $RANK)..."

    kubectl exec $POD_NAME -- bash -c "
        export MASTER_ADDR=$MASTER_ADDR
        export MASTER_PORT=$MASTER_PORT
        export WORLD_SIZE=$WORLD_SIZE
        export RANK=$RANK
        export LOCAL_RANK=0
        export NODE_ID=$NODE_ID
        export CUDA_VISIBLE_DEVICES=0

        cd /workspace

        python3 benchmark.py \
            --batch-size $BATCH_SIZE \
            --seq-length $SEQ_LENGTH \
            --max-steps $MAX_STEPS \
            --gradient-accumulation-steps 1 \
            --lora-rank 16 \
            --num-samples 1000
    " > /tmp/distributed_benchmark_node${NODE_ID}.log 2>&1 &

    echo "  ✓ Started on $POD_NAME (logs: /tmp/distributed_benchmark_node${NODE_ID}.log)"
}

# Launch benchmark on all nodes in parallel
echo "Launching distributed benchmark on all nodes..."
echo ""

for i in $(seq 1 $WORLD_SIZE); do
    run_on_node $i
done

echo ""
echo "Benchmarks launched on all $WORLD_SIZE nodes!"
echo ""
echo "Monitor progress:"
echo "  kubectl logs -f benchmark-node-1  # Master node (shows main progress)"
echo "  kubectl logs -f benchmark-node-2  # Worker node 2"
echo "  # ... etc"
echo ""
echo "Or monitor all logs:"
for i in $(seq 1 $WORLD_SIZE); do
    echo "  tail -f /tmp/distributed_benchmark_node${i}.log"
done
echo ""

# Wait for all background jobs to complete
echo "Waiting for benchmarks to complete..."
wait

echo ""
echo "============================================"
echo "Benchmarks completed!"
echo "============================================"
echo ""

# Collect results
echo "Collecting results from all nodes..."
mkdir -p results/distributed
for i in $(seq 1 $WORLD_SIZE); do
    kubectl cp benchmark-node-$i:/results/distributed/ results/distributed/node${i}/ 2>/dev/null || true
    if [ -d "results/distributed/node${i}" ]; then
        echo "  ✓ Collected from node $i"
    fi
done

echo ""
echo "Results saved to: results/distributed/"
echo ""
echo "Analyze results:"
echo "  python3 analysis/analyze_distributed_results.py"
echo ""

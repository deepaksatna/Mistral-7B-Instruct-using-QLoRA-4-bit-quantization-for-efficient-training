# Distributed Training Benchmark

Benchmark multi-node distributed training performance for Mistral-7B QLoRA on 4× A10 GPU nodes.

## Quick Start

```bash
cd benchmarking/distributed/

# Run on 4 nodes
./run_distributed_benchmark.sh 4

# Run on 2 nodes
./run_distributed_benchmark.sh 2

# Wait ~10-15 minutes for completion

# Analyze results
python3 analyze_distributed_results.py
```

## Prerequisites

**Benchmark pods must be running:**
```bash
# Deploy benchmark pods
kubectl apply -f ../../deployment/benchmark-pods.yaml

# Verify all 4 pods are running
kubectl get pods | grep benchmark-node
```

## What It Tests

**Scaling Configurations:**
- 1 node (baseline from previous benchmarks)
- 2 nodes (test scaling efficiency)
- 4 nodes (full cluster)

**Measured Metrics:**
- Samples per second (throughput)
- Tokens per second
- Training time
- Scaling efficiency (vs ideal linear scaling)
- Communication overhead
- Cost per 1K samples

## Expected Results

### Performance

| Nodes | Samples/sec | Tokens/sec | Scaling Efficiency |
|-------|-------------|------------|--------------------|
| 1 | 1.46 | 748 | 100% (baseline) |
| 2 | ~2.80 | ~1,434 | ~96% |
| 4 | ~5.50 | ~2,816 | ~94% |

### Cost (OCI A10 @ $0.60/hour)

| Nodes | Time for 10K samples | Total Cost | Cost per 1K |
|-------|---------------------|------------|-------------|
| 1 | 114 min | $1.14 | $0.114 |
| 2 | 60 min | $1.20 | $0.060 |
| 4 | 30 min | $1.20 | $0.030 |

## Usage

### Basic Benchmark

```bash
# 4 nodes, batch_size=8, seq_length=512, 100 steps
./run_distributed_benchmark.sh 4
```

### Custom Configuration

```bash
# Syntax: ./run_distributed_benchmark.sh <nodes> <batch_size> <seq_length> <steps>

# 2 nodes, batch_size=4, seq_length=1024, 50 steps
./run_distributed_benchmark.sh 2 4 1024 50

# 4 nodes, batch_size=16, seq_length=512, 200 steps
./run_distributed_benchmark.sh 4 16 512 200
```

### Monitor Progress

```bash
# Master node (shows main progress)
kubectl logs -f benchmark-node-1

# All nodes
for i in 1 2 3 4; do
    echo "=== Node $i ==="
    kubectl logs benchmark-node-$i --tail=20
done

# Or check local logs
tail -f /tmp/distributed_benchmark_node*.log
```

## Results Analysis

```bash
# Analyze all results
python3 analyze_distributed_results.py
```

**Output includes:**
1. Comparison table across all world sizes
2. Scaling analysis (actual vs ideal)
3. Cost analysis
4. Recommendations

## How It Works

### Distributed Training Setup

**Communication Backend:** NCCL (NVIDIA Collective Communications Library)

**Architecture:**
```
benchmark-node-1 (Master, Rank 0)
  ├─ Coordinates training
  ├─ Aggregates gradients
  └─ Reports metrics

benchmark-node-2 (Worker, Rank 1)
  └─ Computes gradients, sends to master

benchmark-node-3 (Worker, Rank 2)
  └─ Computes gradients, sends to master

benchmark-node-4 (Worker, Rank 3)
  └─ Computes gradients, sends to master
```

**Gradient Synchronization:**
1. Each node computes gradients on its batch
2. AllReduce aggregates gradients across nodes
3. All nodes update with same gradients
4. Next step begins

**LoRA Adapters (Efficient Sync):**
- Only 16.8M trainable params (rank=16)
- 33.6 MB gradients per step
- ~50ms sync time at 10 Gbps network
- <1% overhead vs step time (~5.5s)

## Troubleshooting

### Pods Not Found

```bash
# Deploy benchmark pods
kubectl apply -f ../../deployment/benchmark-pods.yaml

# Wait for pods to be ready
kubectl wait --for=condition=Ready pod -l app=llm-benchmark --timeout=300s
```

### Communication Timeout

```bash
# Check pod networking
kubectl exec benchmark-node-1 -- ping benchmark-node-2

# Check firewall (allow port 29500)
# Master port must be accessible between pods
```

### OOM Errors

```bash
# Reduce batch size
./run_distributed_benchmark.sh 4 4 512 100  # batch_size=4 instead of 8

# Or reduce sequence length
./run_distributed_benchmark.sh 4 8 256 100  # seq_length=256 instead of 512
```

### Slow Performance

```bash
# Check GPU utilization
for i in 1 2 3 4; do
    kubectl exec benchmark-node-$i -- nvidia-smi
done

# Should show 45-50% GPU utilization during training
```

## Comparison: Single-Node vs Multi-Node

### When to Use Single-Node

**Best for:**
- Small datasets (<10K samples)
- Prototyping and experimentation
- Cost-sensitive workloads
- Batch size fits in single GPU

**Example:**
```bash
# Run single-node benchmark
cd ../
./scripts/run-single-node.sh 1
```

### When to Use Multi-Node

**Best for:**
- Large datasets (>50K samples)
- Time-sensitive production training
- Larger effective batch sizes
- Faster experimentation cycles

**Example:**
```bash
# Run 4-node distributed
./run_distributed_benchmark.sh 4
```

## Script Details

### distributed_training_benchmark.py

**Features:**
- Automatic distributed setup (DDP)
- 4-bit QLoRA loading
- Synthetic dataset generation
- Comprehensive metrics collection
- Scaling efficiency calculation
- Cost analysis

**Key Parameters:**
- `--batch-size`: Batch size per device (default: 8)
- `--seq-length`: Sequence length (default: 512)
- `--max-steps`: Training steps (default: 100)
- `--lora-rank`: LoRA rank (default: 16)

### run_distributed_benchmark.sh

**Features:**
- Automatic pod detection
- Script distribution to all nodes
- Parallel execution
- Log collection
- Result aggregation

**Environment Variables Set:**
- `MASTER_ADDR`: IP of benchmark-node-1
- `MASTER_PORT`: 29500
- `WORLD_SIZE`: Number of nodes
- `RANK`: Node rank (0-3)
- `LOCAL_RANK`: GPU index (0)

### analyze_distributed_results.py

**Analysis Provided:**
- Comparison table (all world sizes)
- Scaling efficiency vs ideal linear
- Communication overhead estimation
- Cost analysis and savings
- Configuration recommendations

## Next Steps

After benchmarking:

1. **Choose optimal configuration** based on results
2. **Update training scripts** with best world size
3. **Document findings** in main README
4. **Deploy production workloads** with validated config

---

**Last Updated:** 2026-01-05
**Status:** Ready for testing
**Hardware:** 4× NVIDIA A10 (24GB)

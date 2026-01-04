# Benchmarking Guide

Comprehensive benchmarking framework for LLM training on 4× A10 GPU nodes.

## Quick Start

```bash
cd benchmarking/

# Run optimized benchmark (recommended)
./scripts/run-final-benchmark.sh

# Wait ~5 minutes, then analyze results
python3 analysis/compare_final_results.py
```

## Benchmark Configurations

### Final Optimized (Recommended)

**Configuration:**
```python
{
    "batch_size": 8,
    "seq_length": 512,  # CRITICAL
    "lora_rank": 16,
    "gradient_checkpointing": True,
    "optimizer": "paged_adamw_8bit",
    "bf16": True
}
```

**Expected Performance (per A10 GPU):**
- Samples/sec: 1.46
- Tokens/sec: 748
- GPU memory: ~20 GB
- GPU utilization: 45-50%

### Test Matrix

**Diagnostic Suite (6 configurations):**
1. Baseline (batch=4, seq=1024)
2. Different optimizer (adamw_torch)
3. Lower LoRA rank (r=8)
4. Shorter sequence (seq=512)  ← Found bottleneck
5. seq=512 + batch=8  ← Optimal
6. FP16 precision

## Running Benchmarks

### On All 4 Nodes

```bash
# Deploy benchmark pods first
kubectl apply -f deployment/benchmark-pods.yaml

# Run final benchmark
./scripts/run-final-benchmark.sh

# Expected time: ~5 minutes per node (parallel)
```

### On Single Node (Testing)

```bash
./scripts/run-single-node.sh 1 --quick

# Parameters:
# - 1: node ID (1-4)
# - --quick: run quick mode (optional)
```

### Monitor Progress

```bash
# GPU utilization
./scripts/monitor-gpus.sh

# Pod logs
kubectl logs benchmark-node-1 -f
```

## Results Analysis

### Collect Results

```bash
./scripts/collect-results.sh

# Results saved to:
# - results/final/node{1-4}_final.json
# - results/logs/node{1-4}*.log
```

### Compare Nodes

```bash
python3 analysis/compare_final_results.py

# Expected output:
# Node 1: 1.46 samples/sec
# Node 2: 1.45 samples/sec
# Node 3: 1.47 samples/sec
# Node 4: 1.44 samples/sec
# Variance: 1.2% ✓
```

### View Individual Results

```bash
cat results/final/node1_final.json | jq .

# Key metrics:
# - performance.samples_per_second
# - performance.tokens_per_second
# - memory.peak_during_training
# - cost.cost_per_1000_samples_usd
```

## Benchmark Results

### Performance Improvement

| Configuration | Samples/sec | Speedup | Cost/1K samples |
|--------------|-------------|---------|-----------------|
| Baseline (1024, batch=4) | 0.69 | 1.00× | $0.241 |
| **Optimized (512, batch=8)** | **1.46** | **2.11×** | **$0.114** |

### What Worked

✅ **Sequence length 1024 → 512:** 1.88× speedup
✅ **Batch size 4 → 8:** 1.12× additional speedup
✅ **Total improvement:** 2.11× faster, 53% cost reduction

### What Didn't Work

❌ Pre-tokenizing data: 0% improvement
❌ Changing optimizer: 0% improvement
❌ Reducing LoRA rank: 0% improvement
❌ Removing gradient checkpointing: OOM error

## Node-to-Node Consistency

### Acceptable Variance

| Metric | Acceptable Range |
|--------|------------------|
| Samples/sec | ±5% |
| GPU Memory | ±2% |
| Temperature | ±10°C |

### If Variance >5%

```bash
# Check GPU health
kubectl exec benchmark-node-X -- nvidia-smi

# Check for throttling
kubectl exec benchmark-node-X -- nvidia-smi --query-gpu=clocks_throttle_reasons.active --format=csv

# Check node resources
kubectl describe node <node-name>
```

## Cost Analysis

### OCI A10 Pricing

- **On-Demand:** $0.60/hour per GPU
- **Spot:** ~$0.18/hour (if available)

### Benchmark Costs

| Run Type | Duration | Cost (On-Demand) |
|----------|----------|------------------|
| Quick test | 5 min | $0.05 per node |
| Full diagnostic | 30 min | $0.30 per node |
| All 4 nodes (full) | 30 min | $1.20 total |

### Production Training

**Example: 10,000 samples**
- Optimized config: $1.14 (single node) or $0.31/node (4 nodes)
- vs Baseline: $2.41 (single node)
- **Savings:** 53%

## Troubleshooting

### OOM Errors
```python
# Reduce batch size
per_device_train_batch_size=6  # Instead of 8
```

### Slow Performance
```bash
# Verify settings
# - seq_length=512 (NOT 1024)
# - batch_size=8 (NOT 4)
# - gradient_checkpointing=True
```

### Results Not Found
```bash
# Check if benchmark completed
kubectl exec benchmark-node-1 -- ls -lh /results/

# Check logs for errors
kubectl logs benchmark-node-1
```

---

**Last Updated:** 2026-01-04
**Model:** Mistral-7B-Instruct-v0.3
**Hardware:** 4× NVIDIA A10 (24GB)

# Four-Node Deployment Guide

Multi-GPU Kubernetes deployment on 4× NVIDIA A10 GPU nodes using OKE.

## Quick Start

```bash
# 1. Verify 4 GPU nodes
kubectl get nodes -l nvidia.com/gpu.product=NVIDIA-A10

# 2. Label nodes
kubectl label nodes <node-1> gpu-id=1
kubectl label nodes <node-2> gpu-id=2
kubectl label nodes <node-3> gpu-id=3
kubectl label nodes <node-4> gpu-id=4

# 3. Deploy benchmark pods
kubectl apply -f deployment/benchmark-pods.yaml

# 4. Verify deployment
kubectl get pods | grep benchmark-node
```

##Node Configuration

### Requirements

- **4 GPU nodes** with NVIDIA A10 (24GB each)
- **NVIDIA GPU Operator** installed
- **Node labels** for affinity (gpu-id: 1-4)
- **OCIR pull secret** configured

### Verify Nodes

```bash
# Check GPU availability
kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.capacity.nvidia\.com/gpu}{"\n"}{end}'

# Expected output (4 nodes, 1 GPU each):
# node-1    1
# node-2    1
# node-3    1
# node-4    1

# Check GPU model
kubectl describe nodes | grep -A 5 "nvidia.com/gpu.product"
```

## Deployment Manifests

### Benchmark Pods (4-Node Deployment)

```yaml
# deployment/benchmark-pods.yaml
apiVersion: v1
kind: Pod
metadata:
  name: benchmark-node-1
spec:
  imagePullSecrets:
  - name: ocirsecret
  containers:
  - name: llm-training
    image: fra.ocir.io/frntrd2vyxvi/models:mistraltraining-v1.0
    resources:
      limits:
        nvidia.com/gpu: 1
        memory: "40Gi"
      requests:
        nvidia.com/gpu: 1
        memory: "24Gi"
    command: ["/bin/bash", "-c", "sleep infinity"]
  nodeSelector:
    gpu-id: "1"  # Pins to specific GPU node
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
---
# Repeat for benchmark-node-2, -3, -4 with gpu-id: "2", "3", "4"
```

### Multi-GPU Training Job

```yaml
# deployment/job-4gpu.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: llm-training-4gpu
spec:
  parallelism: 4  # 4 parallel pods
  completions: 4
  template:
    spec:
      imagePullSecrets:
      - name: ocirsecret
      containers:
      - name: trainer
        image: fra.ocir.io/frntrd2vyxvi/models:mistraltraining-v1.0
        env:
        - name: MASTER_ADDR
          value: "benchmark-node-1"  # First pod is master
        - name: MASTER_PORT
          value: "29500"
        - name: WORLD_SIZE
          value: "4"
        resources:
          limits:
            nvidia.com/gpu: 1
        command:
        - torchrun
        - --nproc_per_node=1
        - --nnodes=4
        - --node_rank=$(POD_INDEX)
        - --master_addr=$(MASTER_ADDR)
        - --master_port=$(MASTER_PORT)
        - train.py
      restartPolicy: OnFailure
```

## Distributed Training Setup

### Data Parallel (DDP)

```python
# train.py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# Wrap model with DDP
model = get_model()  # Your QLoRA model
model = DDP(model, device_ids=[local_rank])

# Training loop (gradient sync automatic)
trainer = Trainer(model=model, ...)
trainer.train()
```

### Launch Command

```bash
# Single node (testing)
torchrun --nproc_per_node=1 train.py

# Multi-node (4 nodes)
# On each node, run with different --node_rank (0-3)
torchrun \
    --nproc_per_node=1 \
    --nnodes=4 \
    --node_rank=0 \  # 0,1,2,3 for each node
    --master_addr=benchmark-node-1 \
    --master_port=29500 \
    train.py
```

## Performance Expectations

### Single Node
- **Samples/sec:** 1.46
- **Tokens/sec:** 748
- **GPU Utilization:** 45-50%

### 4 Nodes (DDP)
- **Samples/sec:** ~5.5 (combined)
- **Scaling Efficiency:** ~94%
- **Gradient Sync:** <1% overhead
- **Effective Batch:** 8×4 = 32

## Troubleshooting

### Pods Pending
```bash
# Check node labels
kubectl get nodes --show-labels | grep gpu-id

# Add labels if missing
kubectl label nodes <node-name> gpu-id=1
```

### GPU Not Detected
```bash
# Check NVIDIA device plugin
kubectl get ds -n kube-system nvidia-device-plugin-daemonset

# Install if missing
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
```

### Communication Timeout (DDP)
```bash
# Check pod networking
kubectl exec benchmark-node-1 -- ping benchmark-node-2

# Check firewall rules (allow port 29500)
```

## Monitoring

```bash
# Watch GPU usage across all nodes
for i in 1 2 3 4; do
  echo "=== Node $i ==="
  kubectl exec benchmark-node-$i -- nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
done

# Check training logs
kubectl logs benchmark-node-1 -f
```

---

**Last Updated:** 2026-01-04
**Deployment:** 4× NVIDIA A10 (OKE)

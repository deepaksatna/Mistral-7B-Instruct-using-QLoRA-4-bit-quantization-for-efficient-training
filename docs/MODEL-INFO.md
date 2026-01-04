# Model Information and A10 GPU Sizing

Technical details about Mistral-7B and NVIDIA A10 GPU specifications for optimal configuration.

## Model: Mistral-7B-Instruct-v0.3

### Architecture

**Model Specifications:**
- **Parameters:** 7.25 billion
- **Architecture:** Mistral (decoder-only transformer)
- **Layers:** 32
- **Attention Heads:** 32 (GQA with 8 KV heads)
- **Hidden Size:** 4096
- **Vocabulary Size:** 32,000
- **Context Length:** Up to 32,768 (extended RoPE)

**Why Mistral-7B?**
- State-of-the-art performance for 7B size
- Efficient attention mechanism (Grouped Query Attention)
- Fast inference and training
- Good instruction-following capabilities
- Apache 2.0 license (commercial use allowed)

### Memory Requirements

**By Precision:**

| Precision | Parameters Size | Activations (batch=1, seq=1024) | Total (Training) |
|-----------|----------------|--------------------------------|------------------|
| FP32 | 29 GB | ~8 GB | ~40 GB |
| FP16 | 14.5 GB | ~4 GB | ~20 GB |
| 8-bit | 7.25 GB | ~2 GB | ~12 GB |
| **4-bit (QLoRA)** | **3.6 GB** | **~1 GB** | **~6 GB** |

**With Gradient Checkpointing:**
- Activation memory reduced by ~50%
- Allows larger batch sizes
- Required for batch_size >4 on A10

**Optimal for A10 (24 GB):**
```python
{
    "precision": "4-bit",
    "batch_size": 8,
    "seq_length": 512,
    "gradient_checkpointing": True,
    # Memory usage: ~20 GB (leaves 4 GB headroom)
}
```

---

## GPU: NVIDIA A10

### Specifications

**Hardware:**
- **Architecture:** Ampere (SM 8.6)
- **CUDA Cores:** 9,216
- **Tensor Cores:** 288 (3rd gen)
- **Memory:** 24 GB GDDR6
- **Memory Bandwidth:** 600 GB/s
- **TDP:** 150W

**Compute Performance:**
- **FP32:** 31.2 TFLOPS
- **FP16 (Tensor Core):** 125 TFLOPS
- **TF32 (Tensor Core):** 62.5 TFLOPS
- **INT8:** 250 TOPS
- **INT4:** 500 TOPS

### Why A10 for LLM Training?

**Advantages:**
- **Large VRAM (24 GB):** Fits 7B models with QLoRA
- **Tensor Cores:** 4× faster matmul vs FP32
- **Cost-Effective:** Lower TCO than A100
- **Widely Available:** On most cloud platforms
- **Power Efficient:** 150W vs 400W (A100)

**Limitations:**
- Memory bandwidth (600 GB/s) limits long sequences
- Not ideal for >13B models
- Slower than A100 for pure compute

### Comparison to Other GPUs

| GPU | VRAM | Memory BW | FP16 TFLOPS | Cost/Hour (OCI) | Best For |
|-----|------|-----------|-------------|-----------------|----------|
| **A10** | **24 GB** | **600 GB/s** | **125** | **$0.60** | **7B models, cost-effective** |
| A100 (40GB) | 40 GB | 1,555 GB/s | 312 | $2.50 | 13B+ models, max performance |
| A100 (80GB) | 80 GB | 2,039 GB/s | 312 | $4.00 | 30B+ models |
| V100 | 32 GB | 900 GB/s | 125 | $1.20 | 7B models (older) |
| T4 | 16 GB | 320 GB/s | 65 | $0.35 | Inference, small models |

**For Mistral-7B QLoRA:**
- A10 is optimal (best cost/performance)
- A100 is overkill (won't be faster for this workload)
- T4 is too small (can't fit batch_size=8)

---

## QLoRA (4-bit Quantization)

### What is QLoRA?

**QLoRA = Quantized Low-Rank Adaptation**
- Load base model in 4-bit (NF4 quantization)
- Add trainable LoRA adapters (FP16/BF16)
- Train only adapters (0.5% of parameters)

**Memory Savings:**
```
Full fine-tuning (FP16): 14.5 GB (model) + 14.5 GB (gradients) + 14.5 GB (optimizer) = 43.5 GB
QLoRA (4-bit): 3.6 GB (model) + 0.07 GB (adapters) + 0.07 GB (optimizer) = ~4 GB

Savings: 91% memory reduction
```

### QLoRA Components

**1. NF4 Quantization:**
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True  # Quantize quantization constants
)
```

**2. LoRA Adapters:**
```python
LoraConfig(
    r=16,  # Rank (trainable params = r × d × num_modules)
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05
)
```

**3. paged_adamw_8bit:**
```python
TrainingArguments(
    optim="paged_adamw_8bit",  # 8-bit optimizer states
    # Further reduces memory vs 32-bit AdamW
)
```

### LoRA Rank Selection

**Trainable Parameters by Rank:**

| Rank | Trainable Params | Memory | Speed | Quality |
|------|-----------------|---------|-------|---------|
| 8 | 8.4M (0.12%) | 34 MB | Fastest | Good |
| **16** | **16.8M (0.23%)** | **67 MB** | **Fast** | **Best** |
| 32 | 33.6M (0.46%) | 134 MB | Slower | Best |
| 64 | 67.2M (0.93%) | 268 MB | Slowest | Best |

**Recommendation:** Rank 16
- Best speed/quality trade-off
- Higher ranks give minimal improvement
- Lower ranks may underfit

---

## Optimal Sizing for A10

### Memory Breakdown (Optimal Config)

```
Component                    Memory
────────────────────────────────────
Model (4-bit)                4.7 GB
LoRA adapters (r=16)         0.5 GB
Activations (batch=8, seq=512, GC) ~12 GB
Optimizer (8-bit)            ~2 GB
Other (buffers, kernels)     ~1 GB
────────────────────────────────────
TOTAL                        ~20 GB / 24 GB
Headroom                     4 GB (17%)
```

### Why We Can't Go Larger

**Batch size 10:**
- Memory: ~22.5 GB (too close to limit)
- Risk of OOM with fragmentation

**Batch size 12:**
- Memory: ~25 GB → OOM guaranteed

**Without gradient checkpointing:**
- Activations: ~24 GB (just for batch=4)
- Total: ~29 GB → OOM

**Sequence length 1024 (vs 512):**
- Activations: 2× larger
- Can only do batch_size=4
- GPU underutilized

### Maximum Model Size for A10

**With QLoRA + Gradient Checkpointing:**
- **7B models:** Optimal (our use case)
- **13B models:** Possible (batch_size=2-4)
- **20B+ models:** OOM

**Without Quantization:**
- **3B models:** FP16 works
- **7B models:** Requires 8-bit or 4-bit
- **13B+ models:** Doesn't fit

---

## Performance Characteristics

### Compute vs Memory Bound

**Sequence Length Impact:**

| Seq Length | Bottleneck | A10 Utilization | Optimal Batch |
|------------|------------|-----------------|---------------|
| 512 | Compute | 45-50% | 8 |
| 1024 | **Memory BW** | 22% | 4 |
| 2048 | **Memory BW** | 15% | 2 |
| 4096 | **Memory BW** | <10% | 1 |

**Why 512 is Optimal:**
- Balanced compute/memory
- Good GPU utilization
- Allows larger batches

### TF32 Acceleration

**A10 Ampere Architecture Benefits:**
```python
# Enable TF32 for 8× matmul speedup
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# No code changes needed
# Automatic speedup on A10
```

**Impact:**
- FP32 operations run at TF32 speed (62.5 TFLOPS)
- Minimal accuracy impact
- Free performance boost

---

## Cost Analysis

### OCI A10 Pricing

- **GPU.A10.1:** $0.60/hour
- **Per second:** $0.000167
- **Per sample (optimized):** $0.000114

### Training Cost Estimates

**10,000 Sample Fine-tuning (Optimized):**
```
Time: 10,000 / 1.46 samples/sec = 114 minutes
Cost: 114/60 × $0.60 = $1.14
```

**vs Baseline (1024, batch=4):**
```
Time: 10,000 / 0.69 samples/sec = 241 minutes
Cost: 241/60 × $0.60 = $2.41

Savings: $1.27 (53%)
```

**vs Alternatives:**
- OpenAI fine-tuning: $8-12
- AWS p3.2xlarge (V100): $3.06/hour (slower)
- Savings vs OpenAI: 85-95%

---

## Recommendations

### Model Size Selection

**For A10 (24 GB VRAM):**
- **Optimal:** 7B models (our use case)
- **Possible:** 13B models (small batches)
- **Not Recommended:** 20B+ (use A100)

### Precision Selection

**For Training:**
- **Use:** 4-bit (QLoRA) - best memory efficiency
- **Consider:** 8-bit if quality critical
- **Avoid:** FP16/FP32 (waste of memory on 7B)

### Batch Size Selection

**For seq_length=512:**
- **Recommended:** batch_size=8
- **If OOM:** batch_size=6
- **Minimum:** batch_size=4 (suboptimal)

### Sequence Length Selection

**For Most Tasks:**
- **Use:** 512 (optimal performance)
- **Long Context:** 1024 (if needed, but 2× slower)
- **Very Long:** 2048+ (consider A100)

---

**Last Updated:** 2026-01-04
**Model:** Mistral-7B-Instruct-v0.3
**Hardware:** NVIDIA A10 (24GB)
**Precision:** 4-bit QLoRA

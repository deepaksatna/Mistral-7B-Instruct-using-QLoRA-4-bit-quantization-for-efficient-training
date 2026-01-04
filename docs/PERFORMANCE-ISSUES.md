# Performance Issues and Solutions

Comprehensive guide documenting all performance issues encountered and how they were resolved.

## Executive Summary

**Problem:** Training was 2× slower than expected (0.67 samples/sec vs 1.5+ expected)

**Root Cause:** Sequence length (1024) was too long, causing memory bandwidth bottleneck

**Solution:** Reduce sequence length to 512 + increase batch size to 8

**Result:** 2.11× speedup (0.69 → 1.46 samples/sec)

---

## Optimization Journey

### Phase 1: Initial Problem (Slow Training)

**Symptoms:**
- Samples/sec: 0.67
- GPU utilization: 22%
- Step time: 6.0 seconds
- Training 10K samples would take 4+ hours

**Initial Hypothesis:** Data loading bottleneck

### Phase 2: Data Loading Optimization (FAILED)

**Attempt:**
```python
# Pre-tokenize entire dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Use multiple dataloader workers
TrainingArguments(dataloader_num_workers=8)
```

**Expected:** 10-15× speedup
**Actual:** 0% improvement (0.67 → 0.66 samples/sec)

**Learning:** Data loading was NOT the bottleneck

### Phase 3: Gradient Checkpointing Removal (FAILED)

**Attempt:**
```python
# Disable gradient checkpointing
model.gradient_checkpointing_disable()
```

**Expected:** 2-3× speedup
**Actual:** Out of Memory (OOM) even with batch_size=4

**Learning:** Gradient checkpointing is required for A10's 24GB VRAM

### Phase 4: Diagnostic Testing (SUCCESS)

**Method:** Systematically test 6 configurations to isolate bottleneck

**Results:**

| Test | Configuration | Samples/sec | Speedup | Conclusion |
|------|--------------|-------------|---------|------------|
| Baseline | 4bit, batch=4, seq=1024, r=16, GC | 0.69 | 1.00× | Baseline |
| Test 1 | Regular AdamW optimizer | 0.68 | 0.98× | ❌ Optimizer not issue |
| Test 2 | LoRA rank=8 (vs 16) | 0.67 | 0.97× | ❌ LoRA rank not issue |
| **Test 3** | **Seq length=512** | **1.29** | **1.88×** | ✅ **Found bottleneck!** |
| **Test 4** | **Seq=512 + batch=8** | **1.46** | **2.11×** | ✅ **Optimal config** |
| Test 5 | FP16 (vs 4-bit) | 0.78 | 1.13× | ⚠️ Not worth memory cost |

**Key Finding:** Sequence length is the primary bottleneck

---

## Why Sequence Length Matters

### 1. Computational Complexity

**Attention mechanism is O(n²):**
```
Seq 1024: 1024² = 1,048,576 operations per attention head
Seq 512:  512²  =   262,144 operations per attention head

Reduction: 4× fewer operations
```

**For Mistral-7B (32 attention heads × 32 layers):**
- Seq 1024: ~33.5M operations per layer
- Seq 512: ~8.4M operations per layer

### 2. Memory Bandwidth Bottleneck

**A10 Specifications:**
- Memory bandwidth: 600 GB/s
- Compute (FP16): 31.2 TFLOPS

**With long sequences:**
- Attention loads large key/value matrices
- Memory access becomes bottleneck (not compute)
- GPU stalls waiting for data

**Impact:**
```
Seq 1024: High bandwidth usage → GPU underutilized (22%)
Seq 512:  Lower bandwidth usage → GPU better utilized (45-50%)
```

### 3. Gradient Checkpointing Overhead

**How it works:**
- Forward pass: Compute activations, discard most
- Backward pass: Recompute activations (2× compute)

**For long sequences:**
- More activations to recompute
- Overhead scales with sequence length
- Seq 512 has 50% less overhead than seq 1024

---

## All Issues and Fixes

### Issue 1: Slow Training (0.67 samples/sec)

**Solution:** Reduce seq_length from 1024 to 512
```python
TrainingArguments(max_length=512)  # Was 1024
```
**Impact:** 1.88× speedup

### Issue 2: Low GPU Utilization (22%)

**Solution:** Increase batch size from 4 to 8 (after reducing seq_length)
```python
TrainingArguments(
    per_device_train_batch_size=8,  # Was 4
    max_length=512
)
```
**Impact:** GPU utilization → 45-50%, additional 1.12× speedup

### Issue 3: bitsandbytes No GPU Support

**Symptom:**
```
The installed version of bitsandbytes was compiled without GPU support
```

**Solution:**
```dockerfile
ENV BNB_CUDA_VERSION=121
RUN pip install bitsandbytes==0.43.1  # NOT 0.41.3
```

### Issue 4: protobuf Dependency Conflict

**Symptom:**
```
ERROR: protobuf 4.25.1 conflicts with transformers requirement <4.24
```

**Solution:**
```dockerfile
RUN pip install protobuf==4.23.4  # NOT 4.25.1
```

### Issue 5: Docker Build Out of Space

**Symptom:**
```
no space left on device
```

**Solution:**
```bash
# Use shared storage for temp
./build_with_fss.sh /mnt/coecommonfss/tmp

# Or clean Docker
docker system prune -a -f
```

### Issue 6: OOM During Training

**Symptom:**
```
CUDA out of memory. Tried to allocate 2.5 GB
```

**Solution:**
```python
# Ensure gradient checkpointing enabled
model.gradient_checkpointing_enable()

# Reduce batch size if still OOM
per_device_train_batch_size=6  # Instead of 8
```

---

## What We Learned

### 1. Profile Before Optimizing

**Wrong Hypotheses:**
- Data loading bottleneck → 0% improvement
- Gradient checkpointing overhead → Caused OOM
- Optimizer inefficiency → <2% impact
- LoRA configuration → <3% impact

**Right Hypothesis:**
- Sequence length → 88% improvement

**Lesson:** Test systematically, don't assume

### 2. Attention is O(n²)

**Sequence length has exponential impact:**
- 2× longer sequence = 4× more compute
- With gradient checkpointing: 8× more memory access

**For LLM training:**
- Shorter sequences train faster
- Unless you need long context, keep it short
- 512 is optimal for most tasks

### 3. Memory Bandwidth Matters

**GPU specs have two key numbers:**
1. Compute (TFLOPS)
2. Memory bandwidth (GB/s)

**For transformers:**
- Long sequences are memory-bound
- Short sequences are compute-bound
- A10's 600 GB/s limits long-sequence performance

### 4. Trade-offs Exist

**We traded:**
- Memory headroom (5 GB → 20 GB used)
- Sequence length capability (1024 → 512)

**For:**
- 2.11× faster training
- Better GPU utilization
- 53% cost reduction

---

## Optimal Configuration

```python
# FINAL OPTIMIZED SETUP
from transformers import TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig
import torch

# Model loading (4-bit quantization)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# LoRA config
lora_config = LoraConfig(
    r=16,  # Rank doesn't significantly impact speed
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=8,      # OPTIMAL (was 4)
    max_length=512,                      # CRITICAL (was 1024)
    gradient_checkpointing=True,         # Required
    bf16=True,
    optim="paged_adamw_8bit",
    learning_rate=2e-4,
    dataloader_num_workers=4,
)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()
```

---

## Troubleshooting Checklist

### If Training is Slow

1. ✅ Verify `max_length=512` (NOT 1024)
2. ✅ Verify `batch_size=8` (NOT 4)
3. ✅ Verify gradient checkpointing enabled
4. ✅ Check GPU utilization (should be 45-50%)
5. ✅ Check memory usage (should be ~20 GB)

### If Getting OOM

1. ✅ Reduce batch size (8 → 6 → 4)
2. ✅ Keep seq_length=512
3. ✅ Ensure gradient checkpointing ON
4. ✅ Check for memory leaks (restart pod)

### If Variance Between Nodes >5%

1. ✅ Check GPU health: `nvidia-smi`
2. ✅ Check for throttling
3. ✅ Check other workloads on nodes
4. ✅ Verify identical configurations

---

## Future Optimization Opportunities

### 1. Flash Attention 2
- Potential: 1.5-2× speedup
- Requires: transformers upgrade, image rebuild

### 2. Sequence Packing
- Pack multiple short sequences into 512 tokens
- Better utilization for variable-length datasets

### 3. Mixed Batch Training
- Alternate between seq=512 and seq=1024
- 70% speedup with minimal quality loss

---

**Last Updated:** 2026-01-04
**Status:** Optimized and validated
**Speedup Achieved:** 2.11×

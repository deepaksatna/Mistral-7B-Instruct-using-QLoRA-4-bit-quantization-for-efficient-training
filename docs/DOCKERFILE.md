# Dockerfile Creation Guide - LLM Training on OCI A10 GPUs

Complete guide for creating offline Docker images for LLM training workloads on Oracle Cloud Infrastructure.

## Table of Contents

- [Overview](#overview)
- [Multi-Stage Build Process](#multi-stage-build-process)
- [Dockerfile Breakdown](#dockerfile-breakdown)
- [Building the Image](#building-the-image)
- [Pre-Loading Models](#pre-loading-models)
- [Optimization for A10 GPUs](#optimization-for-a10-gpus)
- [Troubleshooting](#troubleshooting)

---

## Overview

### Why Offline Docker Images?

**Benefits:**
- No internet dependency during deployment
- Faster pod startup (no downloads)
- Air-gapped environment support
- Exact version control and reproducibility
- Consistent deployments across all nodes

**Challenges Solved:**
- Dependency conflicts (protobuf, bitsandbytes)
- Model download time (Mistral-7B ~14 GB)
- Package installation errors in production
- Version mismatches between environments

### Image Specifications

```
Base Image:    nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
Final Size:    ~25 GB (uncompressed), ~10 GB (compressed)
Build Time:    15-25 minutes (on standard VM)
Target GPU:    NVIDIA A10 (Ampere architecture)
Python:        3.10
CUDA:          12.1
PyTorch:       2.1.2+cu121
```

---

## Multi-Stage Build Process

The Dockerfile uses a 3-stage build for efficiency and clarity:

```
Stage 1: Base        → Install system dependencies and CUDA tools
Stage 2: Dependencies → Install Python packages (PyTorch, transformers, etc.)
Stage 3: Final       → Copy application code and pre-load models
```

### Why Multi-Stage?

1. **Clarity:** Each stage has a single responsibility
2. **Debugging:** Easy to test each stage independently
3. **Caching:** Docker caches layers for faster rebuilds
4. **Size Optimization:** Can exclude build tools from final image

---

## Dockerfile Breakdown

### Stage 1: Base System

```dockerfile
# Stage 1: Base image with CUDA and Python
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# NVIDIA A10 GPU optimization
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_MODULE_LOADING=LAZY

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libssl-dev \
    ca-certificates \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel
```

**Key Points:**

1. **Base Image:** `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04`
   - Includes CUDA 12.1 (matches A10 requirements)
   - Includes cuDNN 8 (for deep learning acceleration)
   - Ubuntu 22.04 LTS (stable, well-supported)

2. **Environment Variables:**
   - `CUDA_MODULE_LOADING=LAZY`: Faster startup (loads CUDA modules on demand)
   - `NVIDIA_DRIVER_CAPABILITIES=compute,utility`: Required for nvidia-smi and compute

3. **Python 3.10:**
   - Required for compatibility with transformers 4.36.2
   - Newer versions may have dependency conflicts

4. **System Tools:**
   - `build-essential`, `cmake`: For compiling Python packages (bitsandbytes)
   - `git`: For cloning models/code if needed
   - `vim`, `htop`: Debugging tools

### Stage 2: Python Dependencies

```dockerfile
# Stage 2: Install Python dependencies
FROM base AS dependencies

# Install PyTorch with CUDA 12.1 support (A10 compatible)
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Install bitsandbytes with CUDA support
ENV BNB_CUDA_VERSION=121
RUN pip install --no-cache-dir bitsandbytes==0.43.1

# Install Hugging Face ecosystem
RUN pip install --no-cache-dir \
    transformers==4.36.2 \
    accelerate==0.25.0 \
    peft==0.7.1 \
    datasets==2.16.1 \
    sentencepiece==0.1.99 \
    protobuf==4.23.4

# Install experiment tracking
RUN pip install --no-cache-dir \
    wandb==0.16.2 \
    tensorboard==2.15.1

# Install data processing
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    pandas==2.0.3 \
    scipy==1.11.4 \
    scikit-learn==1.3.2

# Install visualization
RUN pip install --no-cache-dir \
    matplotlib==3.7.2 \
    seaborn==0.12.2

# Install utilities
RUN pip install --no-cache-dir \
    pyyaml==6.0 \
    tqdm==4.65.0 \
    psutil==5.9.6 \
    pynvml==11.5.0 \
    gpustat==1.1

# Install development tools
RUN pip install --no-cache-dir \
    ipython==8.18.1 \
    jupyter==1.0.0 \
    black==23.12.1 \
    flake8==7.0.0
```

**Critical Dependencies:**

1. **PyTorch 2.1.2+cu121:**
   - `cu121` = CUDA 12.1 support
   - Must use `--index-url https://download.pytorch.org/whl/cu121`
   - Default PyPI version doesn't have CUDA support

2. **bitsandbytes 0.43.1:**
   - **CRITICAL:** Version 0.41.3 has no GPU support
   - Must set `BNB_CUDA_VERSION=121` environment variable
   - Required for 4-bit quantization (QLoRA)

3. **protobuf 4.23.4:**
   - **CRITICAL:** Version 4.25.1+ conflicts with transformers
   - Must use exactly 4.23.4 for compatibility

4. **transformers 4.36.2:**
   - Supports Mistral-7B architecture
   - Compatible with PEFT 0.7.1 for LoRA

5. **gpustat 1.1:**
   - **CRITICAL:** Version 1.1.1 doesn't exist (causes build failure)
   - Use 1.1 (latest available)

**Why Separate RUN Commands?**
- Each `RUN` creates a Docker layer
- Grouped by functionality for better caching
- If one group fails, others remain cached

### Stage 3: Final Image

```dockerfile
# Stage 3: Final image with models
FROM dependencies AS final

WORKDIR /workspace

# Create necessary directories
RUN mkdir -p \
    /workspace/data \
    /workspace/logs \
    /workspace/results/metrics \
    /workspace/results/checkpoints \
    /workspace/results/plots \
    /models \
    /shared-data

# Copy pre-trained models from build context
# Models should be downloaded to offline/models/ before build
COPY models/ /models/

# Set environment variables for offline mode with pre-loaded models
ENV HF_DATASETS_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_CACHE=/models
ENV HF_HOME=/models
ENV TORCH_HOME=/models

# Verify installations
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}')" && \
    python -c "import transformers; print(f'Transformers: {transformers.__version__}')" && \
    python -c "import bitsandbytes; print(f'bitsandbytes: {bitsandbytes.__version__}')"

# List models in the image
RUN echo "=== Pre-loaded Models ===" && \
    ls -lh /models/ && \
    du -sh /models/

CMD ["/bin/bash"]
```

**Key Points:**

1. **Directory Structure:**
   ```
   /workspace/          # Working directory
   ├── data/           # Training data
   ├── logs/           # Training logs
   └── results/        # Outputs
       ├── metrics/    # JSON/CSV metrics
       ├── checkpoints/# Model checkpoints
       └── plots/      # Visualizations

   /models/            # Pre-loaded models
   └── mistralai--Mistral-7B-Instruct-v0.3/

   /shared-data/       # For mounting shared storage (FSS/NFS)
   ```

2. **Offline Mode Environment Variables:**
   - `HF_DATASETS_OFFLINE=1`: Prevents datasets library from going online
   - `TRANSFORMERS_OFFLINE=1`: Prevents transformers from downloading
   - `HF_HUB_OFFLINE=1`: Prevents Hugging Face Hub access
   - All point to `/models/` directory

3. **Verification Steps:**
   - Import tests ensure packages installed correctly
   - List models to verify they're included
   - Show disk usage for transparency

---

## Building the Image

### Preparation

**1. Download Model (Before Build)**

```bash
cd offline/

# Download Mistral-7B to offline/models/
python download_model.py \
    --model-name "mistralai/Mistral-7B-Instruct-v0.3" \
    --output-dir "./models"

# Verify download
ls -lh models/mistralai--Mistral-7B-Instruct-v0.3/
# Should show: config.json, tokenizer files, pytorch_model.bin files (~14 GB)
```

**2. Verify Directory Structure**

```bash
offline/
├── Dockerfile
├── models/
│   └── mistralai--Mistral-7B-Instruct-v0.3/
│       ├── config.json
│       ├── tokenizer.json
│       ├── pytorch_model-00001-of-00003.bin
│       ├── pytorch_model-00002-of-00003.bin
│       └── pytorch_model-00003-of-00003.bin
├── requirements-offline.txt
└── .dockerignore
```

### Build Command

**Standard Build:**

```bash
cd offline/

docker build \
    -f Dockerfile \
    -t llm-training:latest \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    .

# Expected time: 15-25 minutes
# Expected size: ~25 GB
```

**Build with FSS for Temp Storage:**

```bash
# Use shared storage for temp files (avoids disk space issues)
./build_with_fss.sh /mnt/coecommonfss/llmcore/LLM-training/tmp
```

**Build Script Breakdown:**

```bash
#!/bin/bash
# build_with_fss.sh

TMPDIR=${1:-/mnt/coecommonfss/llmcore/LLM-training/tmp}

echo "Using TMPDIR: $TMPDIR"
mkdir -p "$TMPDIR"

# Set Docker to use FSS for temp
export TMPDIR

# Build with BuildKit for better caching
DOCKER_BUILDKIT=1 docker build \
    -f Dockerfile \
    -t llm-training:latest \
    --progress=plain \
    .

# Optionally save to tar
read -p "Save image to tar.gz? (y/n) " -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker save llm-training:latest | gzip > llm-training-latest.tar.gz
fi
```

### Build Output

Expected output during build:

```
Step 1/X : FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base
 ---> a1b2c3d4e5f6

Step 2/X : ENV DEBIAN_FRONTEND=noninteractive
 ---> Running in 1234567890ab
 ---> 0987654321cd

Step 3/X : RUN apt-get update && apt-get install -y python3.10...
 ---> Running in abcdef123456
...
Get:1 http://archive.ubuntu.com/ubuntu jammy InRelease [270 kB]
...
Successfully installed python3.10
 ---> fedcba987654

...

Step 15/X : RUN pip install torch==2.1.2...
 ---> Running in xyz123abc456
Collecting torch==2.1.2
  Downloading torch-2.1.2+cu121-... (2.3 GB)
...
Successfully installed torch-2.1.2

Step 16/X : RUN pip install bitsandbytes==0.43.1
 ---> Running in 789456123abc
Collecting bitsandbytes==0.43.1
...
Successfully installed bitsandbytes-0.43.1

...

Step 25/X : COPY models/ /models/
 ---> 123abc456def

Step 26/X : RUN python -c "import torch; print(f'PyTorch: {torch.__version__}')"
 ---> Running in abc123def456
PyTorch: 2.1.2+cu121
 ---> 456def789ghi

...

Successfully built abc123def456
Successfully tagged llm-training:latest
```

### Verify Build

```bash
# Check image exists
docker images | grep llm-training
# llm-training    latest    abc123def456    5 minutes ago    25.3GB

# Test Python packages
docker run --rm llm-training:latest python -c "
import torch
import transformers
import bitsandbytes
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')  # False on CPU VM
print(f'Transformers: {transformers.__version__}')
print(f'bitsandbytes: {bitsandbytes.__version__}')
"

# Expected output:
# PyTorch: 2.1.2+cu121
# CUDA available: False  # Normal on CPU VM without --gpus flag
# Transformers: 4.36.2
# bitsandbytes: 0.43.1

# Verify model is included
docker run --rm llm-training:latest ls -lh /models/
# Should show mistralai--Mistral-7B-Instruct-v0.3/

# Check model size
docker run --rm llm-training:latest du -sh /models/
# ~14G    /models/
```

---

## Pre-Loading Models

### Why Pre-Load Models?

**Benefits:**
- Pod startup time: <30 seconds (vs 5-10 minutes with download)
- No internet dependency
- Consistent model version across all deployments
- No rate limiting from Hugging Face Hub

### Download Script

```python
# download_model.py
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

def download_model(model_name, output_dir):
    """Download model and tokenizer for offline use."""

    print(f"Downloading {model_name}...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Download tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(output_dir)

    # Download model (FP16 - will be quantized at load time)
    print("Downloading model (this may take 10-15 minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto"  # FP16
    )
    model.save_pretrained(output_dir)

    print(f"\nModel downloaded successfully to {output_dir}")
    print(f"Size: {sum(os.path.getsize(os.path.join(dirpath, filename)) for dirpath, dirnames, filenames in os.walk(output_dir) for filename in filenames) / 1e9:.2f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--output-dir", default="./models/mistralai--Mistral-7B-Instruct-v0.3")
    args = parser.parse_args()

    download_model(args.model_name, args.output_dir)
```

### Usage

```bash
# Download to offline/models/ before building
python download_model.py \
    --model-name "mistralai/Mistral-7B-Instruct-v0.3" \
    --output-dir "./models/mistralai--Mistral-7B-Instruct-v0.3"

# Expected output:
# Downloading mistralai/Mistral-7B-Instruct-v0.3...
# Downloading tokenizer...
# Downloading model (this may take 10-15 minutes)...
# Model downloaded successfully to ./models/mistralai--Mistral-7B-Instruct-v0.3
# Size: 14.28 GB
```

### Alternative: Use Existing Cache

If you already have the model in Hugging Face cache:

```bash
# Find cache location
python -c "from transformers import file_utils; print(file_utils.default_cache_path)"
# Output: /home/user/.cache/huggingface/hub

# Copy from cache to offline/models/
cp -r ~/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/[hash]/* \
      ./models/mistralai--Mistral-7B-Instruct-v0.3/
```

---

## Optimization for A10 GPUs

### CUDA 12.1 Support

**Why CUDA 12.1?**
- A10 is Ampere architecture (SM 8.6)
- Native support in CUDA 12.1
- Best performance with TensorCore utilization
- Required for bitsandbytes 4-bit quantization

### bitsandbytes Configuration

**Critical for 4-bit Quantization:**

```dockerfile
# Set CUDA version for bitsandbytes compilation
ENV BNB_CUDA_VERSION=121

# Install specific version with GPU support
RUN pip install --no-cache-dir bitsandbytes==0.43.1
```

**Why 0.43.1?**
- Version 0.41.3 has no GPU support
- Version 0.42.0 has bugs with CUDA 12.1
- Version 0.43.1 is first stable release with CUDA 12.1 + GPU support

### Memory Optimization

```dockerfile
# Enable TF32 for better A10 performance
ENV NVIDIA_TF32_OVERRIDE=1

# Lazy CUDA module loading (faster startup)
ENV CUDA_MODULE_LOADING=LAZY
```

### A10-Specific Optimizations

**In Training Code (not Dockerfile):**

```python
import torch

# Enable TF32 (Tensor Float 32) for A10 Ampere GPUs
# Provides ~8× speedup on matmul operations vs FP32
# With minimal accuracy impact
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Set memory fraction (A10 has 24GB)
# Reserve 90% for training, 10% for system
torch.cuda.set_per_process_memory_fraction(0.9, 0)
```

---

## Troubleshooting

### Build Failures

**1. Out of Disk Space**

```
Error: no space left on device
```

**Solution:**
```bash
# Check disk usage
df -h

# Clean Docker cache
docker system prune -a -f

# Use FSS for temp storage
./build_with_fss.sh /mnt/coecommonfss/llmcore/LLM-training/tmp
```

**2. Dependency Conflict: protobuf**

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed
  protobuf 4.25.1 is installed but protobuf<4.24,>=3.20.3 is required by tensorboard
```

**Solution:**
```dockerfile
# Use protobuf 4.23.4 (compatible with both transformers and tensorboard)
RUN pip install --no-cache-dir \
    protobuf==4.23.4  # NOT 4.25.1
```

**3. bitsandbytes No GPU Support**

```
The installed version of bitsandbytes was compiled without GPU support
```

**Solution:**
```dockerfile
# Must set CUDA version and use 0.43.1+
ENV BNB_CUDA_VERSION=121
RUN pip install --no-cache-dir bitsandbytes==0.43.1  # NOT 0.41.3
```

**4. gpustat Not Found**

```
ERROR: Could not find a version that satisfies the requirement gpustat==1.1.1
```

**Solution:**
```dockerfile
# Use 1.1 (not 1.1.1 which doesn't exist)
RUN pip install --no-cache-dir gpustat==1.1  # NOT 1.1.1
```

### Runtime Issues

**1. CUDA Not Available in Container**

```python
>>> import torch
>>> torch.cuda.is_available()
False
```

**On CPU VM:** This is expected. CUDA will be available when deployed to GPU nodes.

**On GPU Node:**
```bash
# Verify GPU is exposed to container
docker run --gpus all --rm llm-training:latest nvidia-smi

# Or in Kubernetes
kubectl exec -it <pod-name> -- nvidia-smi
```

**2. Model Not Found**

```
OSError: mistralai/Mistral-7B-Instruct-v0.3 does not appear to be a valid model
```

**Solution:**
```bash
# Verify model is in image
docker run --rm llm-training:latest ls -lh /models/

# Check environment variables
docker run --rm llm-training:latest env | grep HF_
# Should show:
# HF_HOME=/models
# TRANSFORMERS_CACHE=/models
# HF_DATASETS_OFFLINE=1
```

**3. bitsandbytes Import Error**

```
ImportError: libbitsandbytes_cpu.so: cannot open shared object file
```

**Solution:**
```dockerfile
# Ensure build-essential installed in base stage
RUN apt-get install -y build-essential cmake
```

---

## Advanced: Multi-Model Images

To include multiple models in one image:

```dockerfile
# In final stage
COPY models/mistralai--Mistral-7B-Instruct-v0.3/ /models/mistralai--Mistral-7B-Instruct-v0.3/
COPY models/meta-llama--Llama-2-7b-hf/ /models/meta-llama--Llama-2-7b-hf/
COPY models/tiiuae--falcon-7b/ /models/tiiuae--falcon-7b/

# Update model listing
RUN echo "=== Pre-loaded Models ===" && \
    for model in /models/*/; do \
        echo "- $(basename $model)"; \
        du -sh "$model"; \
    done
```

---

## Summary

### Key Takeaways

1. **Use multi-stage builds** for clarity and efficiency
2. **Pin all versions** for reproducibility
3. **Pre-load models** for fast deployments
4. **CUDA 12.1** required for A10 GPUs
5. **bitsandbytes 0.43.1** required for 4-bit quantization
6. **protobuf 4.23.4** for dependency compatibility

### Quick Reference

```bash
# Download model
python download_model.py --model-name "mistralai/Mistral-7B-Instruct-v0.3"

# Build image
docker build -t llm-training:latest -f Dockerfile .

# Verify
docker run --rm llm-training:latest python -c "import torch, transformers; print('OK')"

# Save for transfer
docker save llm-training:latest | gzip > llm-training-latest.tar.gz
```

---

**Last Updated:** 2026-01-04
**Image Version:** 1.0
**Base:** nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

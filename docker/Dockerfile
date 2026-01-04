# Multi-stage Dockerfile for Offline LLM Training
# Stage 1: Base image with CUDA and Python
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

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
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Stage 2: Install Python dependencies
FROM base AS dependencies

# Copy requirements from offline directory
COPY offline/requirements-offline.txt /tmp/requirements.txt

# Install PyTorch with CUDA 12.1 support
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Install bitsandbytes separately with CUDA support
# Use newer version with better CUDA 12.1 compatibility
ENV BNB_CUDA_VERSION=121
RUN pip install --no-cache-dir bitsandbytes==0.43.1

# Install other dependencies (excluding bitsandbytes which is already installed)
RUN pip install --no-cache-dir -r /tmp/requirements.txt || \
    (grep -v "bitsandbytes" /tmp/requirements.txt > /tmp/requirements-no-bnb.txt && \
     pip install --no-cache-dir -r /tmp/requirements-no-bnb.txt)

# Stage 3: Final image with application code
FROM dependencies AS final

# Set working directory
WORKDIR /workspace

# Copy application code (relative to build context root)
COPY configs /workspace/configs
COPY scripts /workspace/scripts
COPY docs /workspace/docs
COPY envs /workspace/envs
COPY README.md /workspace/

# Create necessary directories
RUN mkdir -p /workspace/data \
    /workspace/logs \
    /workspace/results/metrics \
    /workspace/results/checkpoints \
    /workspace/results/plots \
    /models \
    /shared-data

# Set environment variables for offline mode
ENV HF_DATASETS_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_CACHE=/models
ENV HF_HOME=/models
ENV TORCH_HOME=/models

# Make scripts executable
RUN chmod +x /workspace/scripts/*.py

# Default command
CMD ["/bin/bash"]

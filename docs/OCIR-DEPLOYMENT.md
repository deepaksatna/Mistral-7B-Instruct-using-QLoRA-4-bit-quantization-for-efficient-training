# OCIR Deployment Guide - Offline Docker Image Deployment

Complete guide for building Docker images on CPU VMs and deploying to A10 GPU nodes via Oracle Cloud Infrastructure Registry (OCIR).

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [OCIR Setup](#ocir-setup)
- [Build and Push Workflow](#build-and-push-workflow)
- [Deployment to OKE](#deployment-to-oke)
- [Troubleshooting](#troubleshooting)

---

## Overview

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│              Build VM (CPU-only)                          │
│                                                            │
│  1. Build Docker image with CUDA libraries                │
│  2. Tag for OCIR                                          │
│  3. Push to OCIR                                          │
│                                                            │
│  docker build → docker tag → docker push                  │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│    OCIR (Oracle Cloud Infrastructure Registry)            │
│                                                            │
│  - fra.ocir.io/[tenancy]/models:mistraltraining-v1.0     │
│  - Image storage and distribution                         │
│  - Access control via policies                            │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│              OKE Cluster (4× A10 GPU Nodes)               │
│                                                            │
│  1. Create OCIR pull secret                               │
│  2. Deploy pods with imagePullSecrets                     │
│  3. Kubernetes pulls image from OCIR                      │
│  4. Run on A10 GPUs with NVIDIA runtime                   │
└──────────────────────────────────────────────────────────┘
```

### Why OCIR?

**Benefits:**
- **Centralized Storage:** One image, deploy to any OKE cluster
- **Version Control:** Tag images with versions (v1.0, v1.1, etc.)
- **Access Control:** IAM policies control who can pull/push
- **Regional:** Deploy to regions close to your clusters
- **No Transfer Costs:** Within same region/tenancy

**vs Alternatives:**
- Docker Hub: Rate limits, slower from OCI
- Private registry: Need to manage infrastructure
- Local images: Can't share across nodes/clusters

---

## Prerequisites

### On Build VM

**Required:**
```bash
# 1. Docker installed and running
docker --version
# Docker version 24.0.x or later

# 2. OCI CLI installed and configured
oci --version
# oci CLI version 3.x.x

# 3. Sufficient disk space
df -h /var/lib/docker
# Need 100+ GB for build
```

**Configure OCI CLI:**
```bash
oci setup config
# Follow prompts to configure:
# - User OCID
# - Tenancy OCID
# - Region
# - Generate/specify API key

# Test configuration
oci iam region list
```

### On OKE Cluster

**Required:**
```bash
# 1. kubectl configured
kubectl get nodes

# 2. 4 A10 GPU nodes available
kubectl get nodes -l nvidia.com/gpu.product=NVIDIA-A10 --show-labels

# 3. NVIDIA GPU Operator installed
kubectl get pods -n gpu-operator-resources
```

---

## OCIR Setup

### Step 1: Get Tenancy Namespace

```bash
# Get your tenancy namespace
oci os ns get

# Output example:
# {
#   "data": "frntrd2vyxvi"
# }

# Save this for later use
export TENANCY_NAMESPACE="frntrd2vyxvi"
```

### Step 2: Determine OCIR Endpoint

OCIR endpoints are region-specific:

| Region | OCIR Endpoint |
|--------|---------------|
| US East (Ashburn) | `iad.ocir.io` |
| US West (Phoenix) | `phx.ocir.io` |
| EU Central (Frankfurt) | `fra.ocir.io` |
| UK South (London) | `lhr.ocir.io` |
| Asia Pacific (Mumbai) | `bom.ocir.io` |

```bash
# Set based on your region
export OCI_REGION="us-ashburn-1"
export OCIR_ENDPOINT="iad.ocir.io"

# Or for Frankfurt
export OCI_REGION="eu-frankfurt-1"
export OCIR_ENDPOINT="fra.ocir.io"
```

### Step 3: Create Auth Token

**Via OCI Console:**

1. Navigate to: **Identity → Users → Your User**
2. Click **Auth Tokens** (left sidebar)
3. Click **Generate Token**
4. Description: `OCIR Docker Login`
5. Click **Generate Token**
6. **SAVE THE TOKEN** - it won't be shown again

**Token format:** `ABC123def456GHI789jkl...` (40-60 characters)

**Via OCI CLI:**
```bash
oci iam auth-token create \
    --description "OCIR Docker Login" \
    --user-id <your-user-ocid>

# Output:
# {
#   "data": {
#     "token": "ABC123def456GHI789jkl...",
#     ...
#   }
# }

# Save token
export OCIR_TOKEN="ABC123def456GHI789jkl..."
```

### Step 4: Docker Login to OCIR

```bash
# Username format: <tenancy-namespace>/<oci-username>
export OCI_USERNAME="your.email@oracle.com"
export DOCKER_USERNAME="${TENANCY_NAMESPACE}/${OCI_USERNAME}"

# Login
docker login ${OCIR_ENDPOINT}

# Prompts:
# Username: frntrd2vyxvi/your.email@oracle.com
# Password: [paste auth token]

# Expected output:
# Login Succeeded
```

**Troubleshooting login:**
```bash
# If login fails, try explicit credentials:
echo $OCIR_TOKEN | docker login ${OCIR_ENDPOINT} \
    -u "${DOCKER_USERNAME}" \
    --password-stdin
```

---

## Build and Push Workflow

### Automated Script

Use the provided script for complete automation:

```bash
cd offline/

# Make executable
chmod +x build-and-push-ocir.sh

# Run
./build-and-push-ocir.sh
```

### Script Breakdown

```bash
#!/bin/bash
# build-and-push-ocir.sh

set -e  # Exit on error

# Configuration
export OCI_REGION="us-ashburn-1"
export TENANCY_NAMESPACE="frntrd2vyxvi"  # CHANGE THIS
export OCIR_REPO="models"
export IMAGE_TAG="mistraltraining-v1.0"

# Derived variables
export OCIR_ENDPOINT="${OCI_REGION%%.*}.ocir.io"
export FULL_IMAGE_NAME="${OCIR_ENDPOINT}/${TENANCY_NAMESPACE}/${OCIR_REPO}:${IMAGE_TAG}"

echo "============================================"
echo "Building and Pushing to OCIR"
echo "============================================"
echo "Region:     ${OCI_REGION}"
echo "Endpoint:   ${OCIR_ENDPOINT}"
echo "Image:      ${FULL_IMAGE_NAME}"
echo "============================================"

# Step 1: Build image locally
echo "[1/4] Building Docker image..."
docker build -f Dockerfile -t llm-training:latest .

# Step 2: Tag for OCIR
echo "[2/4] Tagging image for OCIR..."
docker tag llm-training:latest ${FULL_IMAGE_NAME}

# Step 3: Login to OCIR (will prompt for credentials)
echo "[3/4] Logging in to OCIR..."
echo "Username format: ${TENANCY_NAMESPACE}/<oci-username>"
docker login ${OCIR_ENDPOINT}

# Step 4: Push to OCIR
echo "[4/4] Pushing to OCIR (this may take 10-15 minutes)..."
docker push ${FULL_IMAGE_NAME}

echo "============================================"
echo "SUCCESS!"
echo "Image available at:"
echo "  ${FULL_IMAGE_NAME}"
echo "============================================"

# Verification
echo ""
echo "Verify in OCI Console:"
echo "  Developer Services → Container Registry"
echo ""
echo "Or via CLI:"
echo "  oci artifacts container image list --compartment-id <compartment-ocid> --repository-name ${OCIR_REPO}"
```

### Manual Steps (Alternative)

If you prefer manual control:

```bash
# 1. Build image
docker build -f Dockerfile -t llm-training:latest .

# 2. Tag for OCIR
docker tag llm-training:latest \
    fra.ocir.io/frntrd2vyxvi/models:mistraltraining-v1.0

# 3. Login to OCIR
docker login fra.ocir.io
# Username: frntrd2vyxvi/your.email@oracle.com
# Password: [auth token]

# 4. Push image
docker push fra.ocir.io/frntrd2vyxvi/models:mistraltraining-v1.0

# Expected output:
# The push refers to repository [fra.ocir.io/frntrd2vyxvi/models]
# 5f70bf18a086: Pushed
# a3b5c80a4eba: Pushed
# ...
# mistraltraining-v1.0: digest: sha256:abc123... size: 1234
```

### Verify Push

**Via OCI Console:**
1. Navigate to: **Developer Services → Container Registry**
2. Find repository: `models`
3. Click repository to see image tags
4. Verify: `mistraltraining-v1.0` appears with correct size

**Via OCI CLI:**
```bash
# List repositories
oci artifacts container repository list \
    --compartment-id <compartment-ocid> \
    --display-name models

# List images in repository
oci artifacts container image list \
    --compartment-id <compartment-ocid> \
    --repository-name models

# Get specific image details
oci artifacts container image get \
    --image-id <image-ocid>
```

---

## Deployment to OKE

### Step 1: Create OCIR Pull Secret

Kubernetes needs credentials to pull from OCIR.

**Automated Setup:**
```bash
chmod +x setup-ocir-secret.sh
./setup-ocir-secret.sh
```

**Manual Setup:**
```bash
# Get auth token (from Step 3 of OCIR Setup)
export OCIR_TOKEN="ABC123def456GHI789jkl..."

# Create secret
kubectl create secret docker-registry ocirsecret \
    --docker-server=fra.ocir.io \
    --docker-username=frntrd2vyxvi/your.email@oracle.com \
    --docker-password="${OCIR_TOKEN}" \
    --docker-email=your.email@oracle.com

# Verify
kubectl get secret ocirsecret
```

**Verify Secret Works:**
```bash
# Test pull with secret
kubectl run test-pull \
    --image=fra.ocir.io/frntrd2vyxvi/models:mistraltraining-v1.0 \
    --image-pull-policy=Always \
    --overrides='{"spec":{"imagePullSecrets":[{"name":"ocirsecret"}]}}' \
    --rm -it \
    --restart=Never \
    -- echo "Pull successful"

# Expected output:
# Pull successful
# pod "test-pull" deleted
```

### Step 2: Deploy Pods with OCIR Image

**Example Deployment Manifest:**

```yaml
# deployment.yaml
apiVersion: v1
kind: Pod
metadata:
  name: llm-training-a10
  labels:
    app: llm-training
spec:
  # Use OCIR pull secret
  imagePullSecrets:
  - name: ocirsecret

  # Specify container
  containers:
  - name: llm-training
    # OCIR image (full path required)
    image: fra.ocir.io/frntrd2vyxvi/models:mistraltraining-v1.0
    imagePullPolicy: Always  # Or IfNotPresent after first pull

    # GPU request
    resources:
      limits:
        nvidia.com/gpu: 1
        memory: "40Gi"
      requests:
        nvidia.com/gpu: 1
        memory: "24Gi"

    # Command (keep pod running)
    command: ["/bin/bash"]
    args: ["-c", "sleep infinity"]

  # Node selector (optional - for specific GPU nodes)
  nodeSelector:
    nvidia.com/gpu.product: NVIDIA-A10

  # Tolerations (if nodes are tainted)
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
```

**Deploy:**
```bash
kubectl apply -f deployment.yaml

# Watch pod startup
kubectl get pods -w

# Expected:
# NAME               READY   STATUS              RESTARTS   AGE
# llm-training-a10   0/1     ContainerCreating   0          5s
# llm-training-a10   0/1     Running             0          30s
# llm-training-a10   1/1     Running             0          35s
```

**Check Logs:**
```bash
# Pod events
kubectl describe pod llm-training-a10

# Should NOT see:
# - "ImagePullBackOff"
# - "ErrImagePull"
# - "FailedMount"

# Should see:
# - "Successfully pulled image..."
# - "Created container llm-training"
# - "Started container llm-training"
```

### Step 3: Verify GPU Access

```bash
# Access pod
kubectl exec -it llm-training-a10 -- /bin/bash

# Inside pod:
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.xx       Driver Version: 535.xx       CUDA Version: 12.1     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# |   0  NVIDIA A10          On   | 00000000:00:04.0 Off |                    0 |
# +-------------------------------+----------------------+----------------------+

# Test PyTorch
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'GPU name: {torch.cuda.get_device_name(0)}')
"

# Expected output:
# CUDA available: True
# GPU count: 1
# GPU name: NVIDIA A10
```

---

## Troubleshooting

### Build Issues

**1. Docker Build Fails: No Space**

```
Error response from daemon: write /var/lib/docker/...: no space left on device
```

**Solution:**
```bash
# Check space
df -h /var/lib/docker

# Clean Docker
docker system prune -a -f

# Use FSS for temp
./build_with_fss.sh /mnt/fss/tmp
```

**2. Push Fails: Unauthorized**

```
unauthorized: authentication required
```

**Solution:**
```bash
# Re-login to OCIR
docker logout fra.ocir.io
docker login fra.ocir.io

# Or with explicit credentials
echo $OCIR_TOKEN | docker login fra.ocir.io \
    -u "${TENANCY_NAMESPACE}/${OCI_USERNAME}" \
    --password-stdin
```

**3. Push Fails: Name Invalid**

```
denied: requested access to the resource is denied
```

**Solution:**
```bash
# Verify image name format:
# Correct: fra.ocir.io/frntrd2vyxvi/models:v1.0
# Wrong:   fra.ocir.io/models:v1.0  (missing tenancy)
# Wrong:   ocir.io/frntrd2vyxvi/models:v1.0  (missing region)

# Check your tag
docker images | grep ocir
```

### Deployment Issues

**1. ImagePullBackOff**

```
pod/llm-training-a10   0/1   ImagePullBackOff   0   2m
```

**Diagnosis:**
```bash
kubectl describe pod llm-training-a10 | grep -A 10 Events

# Common errors:
# - "unauthorized: authentication required"
# - "manifest unknown"
# - "requested access to the resource is denied"
```

**Solution A: Secret Missing**
```bash
kubectl get secret ocirsecret
# If not found:
./setup-ocir-secret.sh
```

**Solution B: Wrong Image Name**
```bash
# Verify image exists in OCIR
oci artifacts container image list \
    --compartment-id <compartment-ocid> \
    --repository-name models

# Update deployment with correct image name
kubectl edit pod llm-training-a10
```

**Solution C: Auth Token Expired**
```bash
# Generate new auth token in OCI Console
# Delete and recreate secret
kubectl delete secret ocirsecret
kubectl create secret docker-registry ocirsecret \
    --docker-server=fra.ocir.io \
    --docker-username=frntrd2vyxvi/your.email@oracle.com \
    --docker-password="<new-token>"
```

**2. Short Name Mode Enforcing**

```
Error: short name mode is enforcing that images have a fully qualified name
```

**Solution:**
```bash
# Use FULL image path (not shortened)
# Wrong: models:mistraltraining-v1.0
# Correct: fra.ocir.io/frntrd2vyxvi/models:mistraltraining-v1.0
```

**3. Policy Error: Not Authorized**

```
message: "Error: failed to pull image ... not authorized"
```

**Solution:**
```bash
# Check compartment policies (via OCI Console)
# Identity → Policies → check for OCIR policies

# Required policy example:
# Allow group <your-group> to read repos in compartment <compartment>

# Or via CLI:
oci iam policy create \
    --compartment-id <tenancy-ocid> \
    --name ocir-pull-policy \
    --description "Allow pulling from OCIR" \
    --statements '["Allow group <group> to read repos in tenancy"]'
```

### Verification Commands

**Check Everything:**
```bash
# 1. Tenancy namespace
oci os ns get

# 2. Available regions
oci iam region list

# 3. OCIR repositories
oci artifacts container repository list \
    --compartment-id <compartment-ocid>

# 4. Images in repository
oci artifacts container image list \
    --compartment-id <compartment-ocid> \
    --repository-name models

# 5. Secret in Kubernetes
kubectl get secret ocirsecret -o yaml

# 6. Decode secret (verify credentials)
kubectl get secret ocirsecret -o jsonpath='{.data.\.dockerconfigjson}' | base64 -d

# 7. Test image pull
kubectl run test --image=fra.ocir.io/frntrd2vyxvi/models:mistraltraining-v1.0 \
    --overrides='{"spec":{"imagePullSecrets":[{"name":"ocirsecret"}]}}' \
    --rm -it -- bash
```

---

## Best Practices

### Image Naming

**Recommended Convention:**
```
<region>.ocir.io/<tenancy>/<repo>:<tag>

Examples:
fra.ocir.io/frntrd2vyxvi/models:mistraltraining-v1.0
fra.ocir.io/frntrd2vyxvi/models:mistraltraining-v1.1
fra.ocir.io/frntrd2vyxvi/llm-base:pytorch-2.1.2-cuda12.1
```

**Tags:**
- Use semantic versioning: `v1.0`, `v1.1`, `v2.0`
- Include key info: `mistraltraining-v1.0`, `mistral7b-qlora-a10`
- Avoid `latest` in production (use specific versions)

### Security

**Auth Tokens:**
- Create separate tokens for different purposes
- Rotate tokens regularly (every 90 days)
- Never commit tokens to git
- Use OCI Vault for production secrets

**Image Security:**
- Scan images for vulnerabilities before push
- Use minimal base images
- Don't include secrets in images
- Review Dockerfile for security issues

### Cost Optimization

**Storage:**
- OCIR charges for storage (~$0.025/GB/month)
- Delete old/unused images
- Use image lifecycle policies

**Transfer:**
- Within same region: free
- Cross-region: charged (avoid if possible)
- Out to internet: charged (use private endpoints)

---

## Summary

### Complete Workflow Checklist

**Build VM:**
- [ ] OCI CLI configured
- [ ] Auth token generated
- [ ] Docker logged in to OCIR
- [ ] Image built successfully
- [ ] Image tagged for OCIR
- [ ] Image pushed to OCIR
- [ ] Image verified in OCIR

**OKE Cluster:**
- [ ] kubectl configured
- [ ] OCIR pull secret created
- [ ] Deployment manifest updated with OCIR image
- [ ] Pod deployed successfully
- [ ] Image pulled without errors
- [ ] GPU accessible in pod

### Quick Commands

```bash
# Build and Push
cd offline/
./build-and-push-ocir.sh

# Deploy
kubectl create secret docker-registry ocirsecret \
    --docker-server=fra.ocir.io \
    --docker-username=frntrd2vyxvi/user@example.com \
    --docker-password="<token>"

kubectl apply -f deployment.yaml
kubectl get pods -w

# Verify
kubectl exec -it llm-training-a10 -- nvidia-smi
```

---

**Last Updated:** 2026-01-04
**OCIR Endpoint:** fra.ocir.io
**Image:** fra.ocir.io/frntrd2vyxvi/models:mistraltraining-v1.0

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Torch (CUDA 12.1 wheels)
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Training deps
COPY runpod/train/requirements.txt /workspace/runpod/train/requirements.txt
RUN pip3 install --no-cache-dir -r /workspace/runpod/train/requirements.txt

# Copy repo into image (optional). In practice, mount your repo into the pod.
# COPY . /workspace/qwen3-vl-fine-tune

CMD ["bash"]

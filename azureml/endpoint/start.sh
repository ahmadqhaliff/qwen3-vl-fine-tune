#!/usr/bin/env bash
set -euo pipefail

# Azure ML typically injects env vars; keep these defaults overridable.
MODEL_PATH="${MODEL_PATH:-/models/qwen3-vl-8b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Latency-oriented defaults (tune later)
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

# Start vLLM OpenAI-compatible server.
# NOTE: Multimodal support for Qwen3-VL must be validated with the exact vLLM build.
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"

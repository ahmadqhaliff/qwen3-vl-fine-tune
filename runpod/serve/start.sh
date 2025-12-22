#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Optional LoRA adapter directory (PEFT)
LORA_DIR="${LORA_DIR:-}"

# Latency-oriented defaults for low concurrency
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

ARGS=(
  --model "$MODEL_ID"
  --host "$HOST"
  --port "$PORT"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
)

# If you trained a LoRA adapter, mount it and set LORA_DIR.
# vLLM LoRA support varies by version/model; treat this as best-effort.
if [[ -n "$LORA_DIR" ]]; then
  ARGS+=(--enable-lora --lora-modules "adapter=${LORA_DIR}")
fi

python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"

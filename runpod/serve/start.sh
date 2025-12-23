#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Optional LoRA adapter directory (PEFT)
LORA_DIR="${LORA_DIR:-}"

# Serverless-friendly: optionally fetch an adapter tarball at startup.
# Provide ONE of:
# - ADAPTER_TGZ_PATH: path to a local .tgz mounted into the container
# - ADAPTER_URL: https URL to a .tgz
ADAPTER_TGZ_PATH="${ADAPTER_TGZ_PATH:-}"
ADAPTER_URL="${ADAPTER_URL:-}"
ADAPTER_SHA256="${ADAPTER_SHA256:-}"

if [[ -z "$LORA_DIR" ]]; then
  if [[ -n "$ADAPTER_TGZ_PATH" ]]; then
    echo "[serve] Using adapter tgz: ${ADAPTER_TGZ_PATH}"
    mkdir -p /app/adapter
    tar -xzf "$ADAPTER_TGZ_PATH" -C /app/adapter
    # If tar contains outputs/qwen3vl-8b-qlora, prefer that.
    if [[ -d /app/adapter/qwen3vl-8b-qlora ]]; then
      LORA_DIR="/app/adapter/qwen3vl-8b-qlora"
    elif [[ -d /app/adapter/outputs/qwen3vl-8b-qlora ]]; then
      LORA_DIR="/app/adapter/outputs/qwen3vl-8b-qlora"
    else
      LORA_DIR="/app/adapter"
    fi
  elif [[ -n "$ADAPTER_URL" ]]; then
    echo "[serve] Downloading adapter from: ${ADAPTER_URL}"
    mkdir -p /app/adapter
    curl -L --retry 5 --retry-delay 2 -o /app/adapter/adapter.tgz "$ADAPTER_URL"
    if [[ -n "$ADAPTER_SHA256" ]]; then
      echo "${ADAPTER_SHA256}  /app/adapter/adapter.tgz" | sha256sum -c -
    fi
    tar -xzf /app/adapter/adapter.tgz -C /app/adapter
    if [[ -d /app/adapter/qwen3vl-8b-qlora ]]; then
      LORA_DIR="/app/adapter/qwen3vl-8b-qlora"
    elif [[ -d /app/adapter/outputs/qwen3vl-8b-qlora ]]; then
      LORA_DIR="/app/adapter/outputs/qwen3vl-8b-qlora"
    else
      LORA_DIR="/app/adapter"
    fi
  fi
fi

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
  echo "[serve] Enabling LoRA from: ${LORA_DIR}"
  ARGS+=(--enable-lora --lora-modules "adapter=${LORA_DIR}")
fi

python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"

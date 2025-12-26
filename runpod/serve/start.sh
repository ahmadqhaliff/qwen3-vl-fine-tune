#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Guard against misconfiguration where a platform overrides the container command
# and accidentally sets MODEL_ID to a script path like /app/start.sh.
if [[ "$MODEL_ID" == "/app/start.sh" ]] || [[ -f "$MODEL_ID" ]]; then
  echo "[serve] WARN: MODEL_ID looks like a file path ($MODEL_ID). Falling back to Qwen/Qwen3-VL-8B-Instruct."
  MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
fi

# Optional LoRA adapter directory (PEFT)
LORA_DIR="${LORA_DIR:-}"

# Serverless-friendly: optionally fetch an adapter tarball at startup.
# Provide ONE of:
# - ADAPTER_TGZ_PATH: path to a local .tgz mounted into the container
# - ADAPTER_URL: https URL to a .tgz
ADAPTER_TGZ_PATH="${ADAPTER_TGZ_PATH:-}"
ADAPTER_URL="${ADAPTER_URL:-}"
ADAPTER_SHA256="${ADAPTER_SHA256:-}"

# Adapter cache settings (helps avoid re-downloading on container restarts).
ADAPTER_CACHE_DIR="${ADAPTER_CACHE_DIR:-/app/adapter}"
ADAPTER_CACHE_TGZ_PATH="${ADAPTER_CACHE_TGZ_PATH:-${ADAPTER_CACHE_DIR}/adapter.tgz}"
ADAPTER_CACHE_MARKER="${ADAPTER_CACHE_MARKER:-${ADAPTER_CACHE_DIR}/.adapter_extracted.sha256}"

pick_lora_dir() {
  if [[ -d "${ADAPTER_CACHE_DIR}/qwen3vl-8b-qlora" ]]; then
    echo "${ADAPTER_CACHE_DIR}/qwen3vl-8b-qlora"
  elif [[ -d "${ADAPTER_CACHE_DIR}/outputs/qwen3vl-8b-qlora" ]]; then
    echo "${ADAPTER_CACHE_DIR}/outputs/qwen3vl-8b-qlora"
  else
    echo "${ADAPTER_CACHE_DIR}"
  fi
}

if [[ -z "$LORA_DIR" ]]; then
  if [[ -n "$ADAPTER_TGZ_PATH" ]]; then
    echo "[serve] Using adapter tgz: ${ADAPTER_TGZ_PATH}"
    mkdir -p "${ADAPTER_CACHE_DIR}"
    cp -f "$ADAPTER_TGZ_PATH" "${ADAPTER_CACHE_TGZ_PATH}"
    ADAPTER_URL=""
  elif [[ -n "$ADAPTER_URL" ]]; then
    echo "[serve] Downloading adapter from: ${ADAPTER_URL}"
    mkdir -p "${ADAPTER_CACHE_DIR}"
    if [[ -f "${ADAPTER_CACHE_TGZ_PATH}" ]]; then
      echo "[serve] Reusing cached adapter tgz: ${ADAPTER_CACHE_TGZ_PATH}"
    else
      curl -L --retry 5 --retry-delay 2 -o "${ADAPTER_CACHE_TGZ_PATH}" "$ADAPTER_URL"
    fi
  fi

  if [[ -z "$LORA_DIR" ]] && [[ -f "${ADAPTER_CACHE_TGZ_PATH}" ]]; then
    ACTUAL_SHA256=""
    if command -v sha256sum >/dev/null 2>&1; then
      ACTUAL_SHA256="$(sha256sum "${ADAPTER_CACHE_TGZ_PATH}" | awk '{print $1}')"
    fi

    if [[ -n "$ADAPTER_SHA256" ]]; then
      echo "${ADAPTER_SHA256}  ${ADAPTER_CACHE_TGZ_PATH}" | sha256sum -c -
      ACTUAL_SHA256="$ADAPTER_SHA256"
    fi

    SHOULD_EXTRACT=1
    if [[ -n "$ACTUAL_SHA256" ]] && [[ -f "$ADAPTER_CACHE_MARKER" ]]; then
      MARKER_SHA="$(cat "$ADAPTER_CACHE_MARKER" 2>/dev/null || true)"
      if [[ "$MARKER_SHA" == "$ACTUAL_SHA256" ]]; then
        SHOULD_EXTRACT=0
      fi
    fi

    if [[ $SHOULD_EXTRACT -eq 1 ]]; then
      echo "[serve] Extracting adapter tgz into: ${ADAPTER_CACHE_DIR}"
      rm -rf "${ADAPTER_CACHE_DIR}/qwen3vl-8b-qlora" "${ADAPTER_CACHE_DIR}/outputs/qwen3vl-8b-qlora" || true
      tar -xzf "${ADAPTER_CACHE_TGZ_PATH}" -C "${ADAPTER_CACHE_DIR}"
      if [[ -n "$ACTUAL_SHA256" ]]; then
        echo -n "$ACTUAL_SHA256" > "$ADAPTER_CACHE_MARKER"
      fi
    else
      echo "[serve] Adapter already extracted (sha256 match)."
    fi

    LORA_DIR="$(pick_lora_dir)"
  fi
fi

# Latency-oriented defaults for low concurrency
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

# Optional: pass through extra vLLM flags without rebuilding the image.
# Example:
#   VLLM_EXTRA_ARGS="--enforce-eager --disable-log-stats"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

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

if [[ -n "$VLLM_EXTRA_ARGS" ]]; then
  echo "[serve] Extra vLLM args: ${VLLM_EXTRA_ARGS}"
  # Split on spaces (keep this simple; avoid quoting inside VLLM_EXTRA_ARGS)
  read -r -a EXTRA_ARGS <<< "$VLLM_EXTRA_ARGS"
  ARGS+=("${EXTRA_ARGS[@]}")
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_LINE="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -n 1 || true)"
  if [[ -n "$GPU_LINE" ]]; then
    echo "[serve] GPU/driver: ${GPU_LINE}"
  fi
fi

python3 - <<'PY'
import os

try:
    import torch
    torch_v = torch.__version__
    cuda_v = torch.version.cuda
except Exception:
    torch_v = "<unknown>"
    cuda_v = "<unknown>"

try:
    import vllm
    vllm_v = getattr(vllm, "__version__", "<unknown>")
except Exception:
    vllm_v = "<unknown>"

print(f"[serve] torch={torch_v} torch_cuda={cuda_v} vllm={vllm_v}")
PY

VLLM_LOG_PATH="${VLLM_LOG_PATH:-/tmp/vllm.log}"
echo "[serve] vLLM log: ${VLLM_LOG_PATH}"

set +e
python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}" 2>&1 | tee "${VLLM_LOG_PATH}"
STATUS=${PIPESTATUS[0]}
set -e

if [[ $STATUS -ne 0 ]]; then
  if grep -qE "cudaErrorUnsupportedPtxVersion|unsupported toolchain" "${VLLM_LOG_PATH}" 2>/dev/null; then
    cat <<'TXT'
[serve] ERROR: vLLM failed with a CUDA PTX/toolchain incompatibility.
[serve] This commonly happens when a kernel backend (often Marlin / CompressedTensors) was built
[serve] with a CUDA toolchain that your host driver can't run.

[serve] Practical fixes to try:
[serve] - Pin a different vLLM base image (avoid :latest) that matches your host driver/CUDA.
[serve] - Try a non-AWQ base model (FP16/BF16) or a different quant format (e.g., GPTQ) to avoid Marlin.
[serve] - If you must use AWQ on this GPU/driver, rebuild vLLM from source in an image that targets your GPU.

[serve] Tip: Mount a persistent volume to /app/adapter so the 615MB adapter isn't re-downloaded on restarts.
TXT
  fi
fi

exit $STATUS

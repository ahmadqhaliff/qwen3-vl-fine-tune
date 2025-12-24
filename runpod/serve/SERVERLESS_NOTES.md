# RunPod Serverless + vLLM (OpenAI-compatible)

This repo includes a vLLM OpenAI-compatible server container in `runpod/serve/`.

## Important reality check
- A true OpenAI-compatible **HTTP** API is easiest with a **Pod** (always-on) because it avoids cold starts and gives better TTFT.
- RunPod **Serverless** can still work, but cold starts will hurt TTFT and not every serverless mode supports arbitrary inbound HTTP the same way.

If your RunPod Serverless product supports exposing say `PORT=8000` as an HTTP endpoint for the container, then this image works directly.

## Common failure: model becomes `/app/start.sh`
If you see logs like:
- `OSError: It looks like the config file at '/app/start.sh' is not a valid JSON file.`

It means the vLLM server is trying to load the model from `/app/start.sh` (a script) instead of Hugging Face.

Fix:
- Leave **Container start command** empty (don’t override the image entrypoint)
- Set `MODEL_ID=Qwen/Qwen3-VL-8B-Instruct`
- In RunPod’s **Model** field, enter `Qwen/Qwen3-VL-8B-Instruct` (not `/app/start.sh`)

## Container
- Dockerfile: `runpod/serve/Dockerfile`
- Entrypoint: `runpod/serve/start.sh`
- Server: `vllm.entrypoints.openai.api_server`

## Adapter options
Provide one of:
- `LORA_DIR`: path to a mounted adapter folder (contains `adapter_model.safetensors` + `adapter_config.json`)
- `ADAPTER_TGZ_PATH`: path to a mounted `qwen3vl-8b-qlora-adapter.tgz`
- `ADAPTER_URL`: https URL to download that tgz at startup (optional `ADAPTER_SHA256`)

### Hosting the adapter on Hugging Face (recommended)
1) Create a Hugging Face repo (model or dataset repo works):
- e.g. `yourname/qwen3vl-8b-doc-extract-lora`

2) Upload the file `qwen3vl-8b-qlora-adapter.tgz` to the repo root.

From a machine with the file:
```bash
pip install -U "huggingface_hub>=0.24.0"
huggingface-cli login
huggingface-cli repo create yourname/qwen3vl-8b-doc-extract-lora --type model
huggingface-cli upload yourname/qwen3vl-8b-doc-extract-lora qwen3vl-8b-qlora-adapter.tgz qwen3vl-8b-qlora-adapter.tgz
```

3) Use this direct-download URL as `ADAPTER_URL`:
```text
https://huggingface.co/yourname/qwen3vl-8b-doc-extract-lora/resolve/main/qwen3vl-8b-qlora-adapter.tgz
```

If your HF repo is private, your serverless container must be able to authenticate to download it (simplest: make the adapter repo public).

## Client (OpenRouter/OpenAI style)
Point your existing OpenRouter/OpenAI client to your endpoint:
- `base_url`: `https://<your-runpod-endpoint>/v1`
- `api_key`: whatever you set

The API paths are OpenAI-style:
- `POST /v1/chat/completions`
- `POST /v1/completions`

## Suggested env
- `MODEL_ID=Qwen/Qwen3-VL-8B-Instruct`
- `MAX_MODEL_LEN=4096`
- `MAX_NUM_SEQS=8`
- `GPU_MEMORY_UTILIZATION=0.90`

## Latency tuning (practical)
End-to-end latency is usually dominated by:
- prompt length (your extraction instructions are long)
- image size (base64 payload + vision encode)
- cold starts (serverless)

Recommended knobs for lower p90 on a single-user/low-concurrency endpoint:
- Reduce prompt tokens: keep rules concise; avoid repeating examples; reuse the same prefix to benefit from prefix cache.
- Reduce image size: resize to ~1024–1280px long side; JPEG quality ~80–90.
- Reduce output tokens: set `max_tokens` to what you truly need (e.g. 400–900).
- Prefer `MAX_NUM_SEQS=1` (or 2) for lowest single-request latency.
- Prefer smaller `MAX_MODEL_LEN` if your prompt+output fits (e.g. 2048–3072).
- Use streaming (`stream=true`) to reduce perceived latency.

You can also pass extra vLLM flags via env:
- `VLLM_EXTRA_ARGS=--disable-log-stats`

Example serverless env (with HF-hosted adapter):
- `ADAPTER_URL=https://huggingface.co/yourname/qwen3vl-8b-doc-extract-lora/resolve/main/qwen3vl-8b-qlora-adapter.tgz`

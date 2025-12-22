# Azure ML training runbook (Qwen3-VL-8B, page-image extraction)

This repo currently prepares **SFT JSONL**. This doc tells you what to do next on Azure ML.

## 1) Split the dataset

```powershell
python scripts\split_jsonl.py --in data\train.jsonl --out-dir data\splits
python scripts\validate_jsonl.py data\splits\train.jsonl
python scripts\validate_jsonl.py data\splits\val.jsonl
python scripts\validate_jsonl.py data\splits\test.jsonl
```

## 2) Pick training method (cost vs speed)

### Recommended starting point: QLoRA (4-bit) + LoRA adapters
- Best cost/VRAM efficiency.
- Lets you train on mid-tier GPUs.

When you have more budget and want faster wall-clock training:
- Switch to **LoRA BF16/FP16** on a bigger GPU (less quant overhead).

## 3) Azure ML compute guidance (practical)

You didn’t pick a SKU yet, so use this decision rule:
- If you want **lowest cost** first: start with a **24GB VRAM GPU** class.
  - If it OOMs at your image resolution, move up.
- If you want **fast iteration** + fewer OOM surprises: start with **~48GB VRAM** class.

For serving (managed online endpoint) targeting low latency:
- Start FP16/BF16, then move to **4-bit (AWQ/GPTQ)** if quality holds.

## 4) Training stack (what to use)

For multimodal Qwen3-VL training, the most reliable options are:
- Qwen’s official Qwen-VL fine-tuning code for the specific Qwen3-VL release
- A Hugging Face Transformers + PEFT training script that explicitly supports Qwen3-VL vision inputs

Unsloth is excellent for many **text-only** LLMs, but multimodal support varies; treat it as optional unless you confirm Qwen3-VL image SFT is supported.

## 5) Hyperparameters (starter)
- epochs: 1–3 (start 1)
- lr: 1e-4 (QLoRA) or 2e-4 (LoRA)
- lora_r: 16
- lora_alpha: 32
- lora_dropout: 0.05
- max_output_tokens: 256–512 (helps latency)

## 6) After training: deploy on Azure ML managed endpoint
- Export adapters (or merged weights)
- Use a fast server (vLLM/SGLang/TensorRT-LLM) depending on model support
- Enable warmup + min instances >= 1

See also:
- `docs/LATENCY_UNDER_1S.md`
- `azureml/endpoint/` (container template)

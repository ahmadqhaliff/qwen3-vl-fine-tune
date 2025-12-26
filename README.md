# Qwen3-VL-8B Fine-tune → Merge → vLLM 4-bit

This repo contains only the **fine-tuning pipeline** for Qwen/Qwen3-VL-8B-Instruct (QLoRA) and the post-training steps to:
1) merge the LoRA adapter back into the base model, then
2) export a **vLLM-loadable 4-bit** checkpoint (BitsAndBytes).

## What you should have after fine-tuning
- A LoRA adapter artifact (in this repo: `qwen3vl-8b-qlora-adapter.tgz`).

## Install

```powershell
python -m pip install -r requirements.txt
```

## Fine-tune (QLoRA)

Training script: `training/train_qwen3vl_qlora.py`

## Merge adapter into base

```powershell
python scripts\merge_adapter_into_base.py --base Qwen/Qwen3-VL-8B-Instruct --adapter qwen3vl-8b-qlora-adapter.tgz --out merged-qwen3vl-8b
```

## Quantize merged model to 4-bit (BitsAndBytes, vLLM-friendly)

This exports a **pre-quantized BitsAndBytes 4-bit** HF checkpoint that vLLM can load.

```powershell
python scripts\quantize_merged_to_4bit_bnb.py --in merged-qwen3vl-8b --out merged-qwen3vl-8b-bnb4 --trust-remote-code
```

In vLLM, you can load the exported checkpoint (in your Linux/vLLM environment):

```bash
vllm serve merged-qwen3vl-8b-bnb4 --trust-remote-code
```

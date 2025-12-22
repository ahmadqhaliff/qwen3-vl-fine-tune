# Training Qwen3-VL-8B for Page-Image PDF Extraction (LoRA/QLoRA)

This doc is an **A→Z** practical guide to fine-tune **Qwen3-VL-8B** for **page-image-based PDF extraction**.

## 0) Target outcome
- Input: PDF page image (rendered) + instruction
- Output: **strict JSON** (no prose)
- Serving: **Azure ML managed online endpoint**
- Latency goal: **< 1s** is sometimes achievable for 8B, but depends on GPU tier, image size, max tokens, and network.

## 1) Choose the tuning method

### Recommended default: SFT + LoRA
Use LoRA if you can train in BF16/FP16 with enough VRAM.
- Pros: faster than QLoRA; strong quality
- Cons: needs more GPU memory

### If VRAM is limited: SFT + QLoRA (4-bit)
- Pros: fits smaller GPUs
- Cons: slower training; sometimes slightly worse quality

## 2) Data format
Use the JSONL chat format described in `schemas/qwen_vl_pdf_extract.schema.json`.

Key rules for extraction quality + speed:
- Keep the **assistant** message as **ONLY JSON**.
- Keep schemas stable (don’t vary key names between samples).
- Prefer short instructions.

## 3) What to fine-tune on (recipe)
For PDF extraction, you get the best returns by training on:
- Your real layouts (same vendors/templates)
- Hard negatives (missing fields, rotated scans, stamps)
- Tables if required (line-items)

Recommended split:
- 80% train / 10% val / 10% test

## 4) Training stack (recommended)
Because multimodal training wiring differs by model release, you should use one of:
- Qwen’s official Qwen-VL fine-tuning code (preferred, if available for Qwen3-VL)
- A Hugging Face Transformers + PEFT training script that supports Qwen3-VL vision inputs

This repo currently provides **configs and constraints**, and we’ll drop in the exact training script once you share a few sample PDFs + your target output schema.

## 5) Hyperparameters (starting point)
These are safe starting ranges for extraction SFT:
- epochs: 1–3 (start with 1)
- effective batch size: 32–128 (via grad accumulation)
- lr (LoRA): 1e-4 to 2e-4
- lr (QLoRA): 1e-4 to 1.5e-4
- warmup ratio: 0.03
- max output tokens: 256–768 (keep small for latency)

LoRA modules (typical):
- attention projections: q_proj, k_proj, v_proj, o_proj
- optionally MLP: up_proj, down_proj, gate_proj

LoRA ranks:
- r=8 or 16 (start with 16 if you can)
- alpha=16–32
- dropout=0.0–0.05

## 6) Practical training speedups
- Use BF16 if supported.
- Use gradient checkpointing only if you must (it slows training).
- Pre-render images and keep them on fast storage.
- Use smaller image sizes that still preserve text readability.

## 7) Memory expectations (rule-of-thumb)
Exact VRAM depends on sequence lengths and image resolution.
- 8B LoRA BF16: often needs **~24–48GB** class GPUs for comfortable training.
- 8B QLoRA 4-bit: can fit smaller (often **~16–24GB**), but not guaranteed for high-res images.

## 8) Next: once you provide data
When you provide sample PDFs + target JSON schema, we will add:
- PDF→image rendering pipeline
- JSONL generation at scale
- A concrete training script + accelerate/deepspeed config
- Eval harness (field-level accuracy)

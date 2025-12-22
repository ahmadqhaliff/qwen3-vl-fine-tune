# Qwen3-VL PDF Extraction Fine-tune (Scaffold)

This repo is a lightweight scaffold for preparing a **page-image-based PDF extraction** dataset for Qwen-VL-style SFT (LoRA/QLoRA), plus basic JSONL validation.

## Recommendation (your current constraints)

### Model choice: Qwen3-VL-8B vs Qwen3-VL-32B-A3B
- **Qwen3-VL-8B**: materially **lighter/faster/cheaper** to serve; usually the best first choice for **structured extraction** if you constrain output to a strict JSON schema.
- **Qwen3-VL-32B-A3B**: higher ceiling on hard layouts (dense tables, messy scans), but **more expensive** and typically **slower** unless you deploy strong GPUs and optimize serving.

For your goal (replace ~7s end-to-end hosted latency), the biggest win is **self-hosting close to your app** + a fast engine. In practice, **8B + good data + strict JSON** often meets extraction needs.

### Fine-tuning method (best default)
- Start with **SFT + LoRA** (if you have enough VRAM) or **SFT + QLoRA (4-bit)** (if VRAM is limited).
- Train on **page images** + **instruction** + **gold JSON**.

### Serving for low latency on Azure
- Fast path: **Azure GPU VM / Azure ML online endpoint / AKS** + **vLLM**.
- Choose GPU based on latency target:
  - 8B: often OK on **A10/L4/L40S** class.
  - 32B: usually needs **A100/H100/L40S** and/or multi-GPU tensor-parallel.

### Quantization (for smaller + cheaper deployment)
Yes. Common options:
- **AWQ 4-bit** (good speed/memory tradeoff)
- **GPTQ 4-bit** (good compatibility)
- **FP8** (if supported by your serving stack + hardware; great speed on newer GPUs)

## Dataset format (JSONL)
Each line is one training example in a multimodal chat format:
- `messages[].content[]` is a list of items: `{ "type": "image", "image": "path" }` and `{ "type": "text", "text": "..." }`.
- The assistant output should be **ONLY** the target JSON (as a string).

See:
- `schemas/qwen_vl_pdf_extract.schema.json`
- `examples/example.jsonl`

## Quick start (validation only)

```powershell
python scripts\make_example_jsonl.py
python scripts\validate_jsonl.py examples\example.jsonl
```

## Docs
- Training: `docs/TRAINING_QWEN3_VL_8B.md`
- Serving/quantization: `docs/QUANTIZATION_AND_SERVING.md`
- Latency checklist: `docs/LATENCY_UNDER_1S.md`
- Azure ML endpoint template: `azureml/README.md`

## Build training JSONL from your `results.json`

1) Extract the bundled images from `data/combined.zip`:

```powershell
python scripts\\extract_combined_zip.py --zip data\\combined.zip
```

If you have additional zips (e.g. `data/bls_another.zip`), extract both:

```powershell
python scripts\\extract_combined_zip.py --zip data\\combined.zip --zip data\\bls_another.zip
```

2) Build JSONL (one record per PDF/document):

```powershell
python scripts\\build_jsonl_from_results.py --images-dir data\\raw\\combined --images-dir data\\raw\\bls_another --limit 50
python scripts\\validate_jsonl.py data\\train.jsonl
```

## Next steps (train/val/test)

```powershell
python scripts\\verify_image_paths.py --in data\\train.jsonl
python scripts\\dataset_stats.py --in data\\train.jsonl
python scripts\\split_jsonl.py --in data\\train.jsonl --out-dir data\\splits
```

Azure ML training notes: `docs/AZUREML_TRAINING_RUNBOOK.md`

## RunPod (A6000) runbook
`runpod/TRAIN_SERVE_RUNBOOK.md`

## Next inputs I’ll need from you (to complete A→Z)
1) Azure plan: **which service** (VM, AKS, Azure ML endpoint) and rough budget.
2) Latency target (p50/p95) and max output tokens.
3) Extraction target schema(s): invoices? bank statements? forms? tables?

Once you provide raw PDFs + labels (or desired fields), we’ll implement:
- PDF → page images (and optionally OCR/text-layer)
- JSONL generation at scale
- Training configs (LoRA/QLoRA) + eval
- Quantization + vLLM deployment recipe

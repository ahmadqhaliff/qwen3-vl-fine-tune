# RunPod (A6000) Train + Serve: Qwen/Qwen3-VL-8B-Instruct

This runbook assumes:
- You use a **RunPod A6000 48GB** pod
- You mount your repo into the pod (e.g. `/workspace/qwen3-vl-fine-tune`)

## 0) Upload your dataset to the pod
The repo intentionally does **not** commit large dataset artifacts (images, zips, JSONL splits).
After `git clone`, you must upload them to the pod.

Minimum required for training:
- `data/splits/train.jsonl` and `data/splits/val.jsonl`
- the referenced images under `data/raw/...`

From your Windows machine (PowerShell), upload via `scp` over the pod's exposed TCP port:

```powershell
# adjust these 3 values
$KEY = "C:\Users\<you>\runpod"
$POD_HOST = "38.147.83.18"
$PORT = 49194

# create folders on the pod
ssh -i $KEY -p $PORT root@$POD_HOST "mkdir -p /root/qwen3-vl-fine-tune/data/splits /root/qwen3-vl-fine-tune/data/raw"

# upload splits
scp -i $KEY -P $PORT -r .\data\splits root@${POD_HOST}:/root/qwen3-vl-fine-tune/data/

# upload extracted images (can be large)
scp -i $KEY -P $PORT -r .\data\raw root@${POD_HOST}:/root/qwen3-vl-fine-tune/data/
```

Alternative (recommended if you have the zips locally): upload `data/*.zip`, then extract/build on the pod.

## 1) Convert dataset splits into simple SFT JSONL
Inside the repo:

```bash
python training/convert_splits_to_sft_jsonl.py --in data/splits/train.jsonl --out data/sft/train.simple.jsonl
python training/convert_splits_to_sft_jsonl.py --in data/splits/val.jsonl --out data/sft/val.simple.jsonl
```

The output format is per-line:
```json
{"id":"...","image":"path/to.png","prompt":"...","response":"{...json...}"}
```

## 2) Train QLoRA (recommended first)

```bash
python training/train_qwen3vl_qlora.py \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --train data/sft/train.simple.jsonl \
  --val data/sft/val.simple.jsonl \
  --out outputs/qwen3vl-8b-qlora \
  --epochs 1 \
  --batch 1 \
  --grad-accum 16 \
  --lr 1e-4 \
  --image-max-side 1536 \
  --max-len 4096
```

If you OOM:
- reduce `--image-max-side` (1024)
- reduce `--max-len`
- increase `--grad-accum` and keep batch=1

## 3) Serve (baseline)

Start with base model (no adapter) to confirm the endpoint works.
Then add LoRA adapter.

We provide a container template in `runpod/serve/`.

## 4) Quantize (optional)
After you have a baseline working, test AWQ/GPTQ for serving.
Quantization support depends on your chosen serving stack.

## Notes
- Your output can be large (50+ containers) so latency is largely **token-bound**.
- Consider splitting into two calls (header fields vs container list) if you need <1s p95.

# RunPod (A6000) Train + Serve: Qwen/Qwen3-VL-8B-Instruct

This runbook assumes:
- You use a **RunPod A6000 48GB** pod
- You mount your repo into the pod (e.g. `/workspace/qwen3-vl-fine-tune`)

## Persistence note (why your data disappeared)
On many RunPod templates, anything under `/root` is on the container/root filesystem and can be **ephemeral** (lost when the pod is stopped/restarted or moved).

To keep datasets, checkpoints, and logs, store them on the attached volume, typically mounted at `/workspace`.

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
ssh -i $KEY -p $PORT root@$POD_HOST "mkdir -p /workspace/qwen3-vl-fine-tune/data/splits /workspace/qwen3-vl-fine-tune/data/raw"

# upload splits
scp -i $KEY -P $PORT -r .\data\splits root@${POD_HOST}:/workspace/qwen3-vl-fine-tune/data/

# upload extracted images (can be large)
scp -i $KEY -P $PORT -r .\data\raw root@${POD_HOST}:/workspace/qwen3-vl-fine-tune/data/
```

If you already cloned into `/root/qwen3-vl-fine-tune`, migrate once:

```bash
mkdir -p /workspace/qwen3-vl-fine-tune
rsync -a --info=progress2 /root/qwen3-vl-fine-tune/ /workspace/qwen3-vl-fine-tune/
mv /root/qwen3-vl-fine-tune /root/qwen3-vl-fine-tune.bak || true
ln -s /workspace/qwen3-vl-fine-tune /root/qwen3-vl-fine-tune
```

Alternative (recommended if you have the zips locally): upload `data/*.zip`, then extract/build on the pod.

## 0.5) Install training dependencies
Inside the pod, from the repo root:

```bash
python3 -m pip install -U pip
python3 -m pip install -r runpod/train/requirements.txt
```

If you see an error mentioning `AutoVideoProcessor requires the Torchvision library`, it means `torchvision` is missing. It is included in `runpod/train/requirements.txt`.

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

Tip: for persistence, ensure you run this from `/workspace/qwen3-vl-fine-tune` so `outputs/` lands on the volume.

If you OOM:
- reduce `--image-max-side` (1024)
- reduce `--max-len`
- increase `--grad-accum` and keep batch=1

If it "moves" but is very slow:
- With `--batch 1 --grad-accum 16`, each progress-bar step is **16 examples**, so minutes/step can be normal.
- The first 1-2 steps are often slow (kernel compile / first-batch preprocessing).
- To speed up wall time, reduce compute: `--image-max-side 1024` and/or `--max-len 2048`.
- If you have VRAM headroom (A6000 48GB), you can try disabling gradient checkpointing (faster): add `--no-grad-checkpointing`.
- If CPU is pegged while GPU is underutilized, increase dataloader parallelism: add `--num-workers 4`.

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

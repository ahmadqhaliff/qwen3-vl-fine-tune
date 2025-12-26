"""Merge a PEFT LoRA adapter back into the base Qwen3-VL model.

Supports adapter provided as:
- a directory containing adapter_config.json
- a .tgz (like qwen3vl-8b-qlora-adapter.tgz) that contains the adapter folder

Example (PowerShell):
  python scripts\merge_adapter_into_base.py --base Qwen/Qwen3-VL-8B-Instruct --adapter qwen3vl-8b-qlora-adapter.tgz --out merged-qwen3vl-8b

Notes:
- This loads the full base model, so run it on a machine with enough RAM/VRAM.
- Uses trust_remote_code by default because Qwen3-VL is typically remote-code.
"""

from __future__ import annotations

import argparse
import os
import tarfile
import tempfile
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, AutoTokenizer


def _find_adapter_dir(root: Path) -> Path:
    # Prefer exact expected folder name if present.
    for candidate in (
        root / "qwen3vl-8b-qlora",
        root / "outputs" / "qwen3vl-8b-qlora",
        root,
    ):
        if (candidate / "adapter_config.json").is_file():
            return candidate

    # Otherwise search.
    for p in root.rglob("adapter_config.json"):
        return p.parent

    raise FileNotFoundError(f"Could not find adapter_config.json under: {root}")


def _load_vision2seq_model(
    model_id_or_path: str,
    *,
    dtype: torch.dtype,
    device_map: str,
    trust_remote_code: bool,
):
    # Keep imports local to avoid import errors on older transformers.
    from transformers import AutoModelForVision2Seq

    return AutoModelForVision2Seq.from_pretrained(
        model_id_or_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base model id or local path")
    ap.add_argument(
        "--adapter",
        required=True,
        help="Adapter directory OR adapter tgz path (e.g. qwen3vl-8b-qlora-adapter.tgz)",
    )
    ap.add_argument("--out", required=True, help="Output directory for merged model")
    ap.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="dtype to load base model with (merge happens in this dtype)",
    )
    ap.add_argument(
        "--device-map",
        default="auto",
        help="transformers device_map (default: auto)",
    )
    ap.add_argument(
        "--trust-remote-code",
        default="1",
        choices=["0", "1"],
        help="trust_remote_code (default: 1)",
    )

    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    trust_remote_code = args.trust_remote_code == "1"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter_path = Path(args.adapter)

    tmp_dir_obj = None
    try:
        if adapter_path.is_file() and adapter_path.suffix in {".tgz", ".gz"}:
            tmp_dir_obj = tempfile.TemporaryDirectory(prefix="adapter_extract_")
            extract_root = Path(tmp_dir_obj.name)
            with tarfile.open(adapter_path, "r:gz") as tf:
                tf.extractall(path=extract_root)
            adapter_dir = _find_adapter_dir(extract_root)
        else:
            adapter_dir = _find_adapter_dir(adapter_path)

        print(f"[merge] base: {args.base}")
        print(f"[merge] adapter_dir: {adapter_dir}")
        print(f"[merge] out: {out_dir}")

        # Load tokenizer/processor first so we can save them even if merge fails later.
        processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=trust_remote_code)
        tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=trust_remote_code)

        base_model = _load_vision2seq_model(
            args.base,
            dtype=dtype,
            device_map=args.device_map,
            trust_remote_code=trust_remote_code,
        )

        peft_model = PeftModel.from_pretrained(
            base_model,
            str(adapter_dir),
            is_trainable=False,
        )

        merged = peft_model.merge_and_unload()

        # Ensure eval mode before save.
        merged.eval()

        merged.save_pretrained(out_dir, safe_serialization=True, max_shard_size="5GB")
        tokenizer.save_pretrained(out_dir)
        processor.save_pretrained(out_dir)

        # Carry over any run_info.json if present.
        run_info = adapter_dir / "run_info.json"
        if run_info.is_file():
            (out_dir / "adapter_run_info.json").write_text(run_info.read_text(encoding="utf-8"), encoding="utf-8")

        print("[merge] done")

    finally:
        if tmp_dir_obj is not None:
            tmp_dir_obj.cleanup()


if __name__ == "__main__":
    # Make HF downloads a bit more predictable on shared boxes.
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    main()

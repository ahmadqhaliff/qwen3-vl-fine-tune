"""Best-effort QLoRA SFT trainer for Qwen/Qwen3-VL-8B-Instruct.

Notes:
- Multimodal fine-tuning support depends on your installed `transformers` version.
- This script is designed for a RunPod Linux GPU environment (A6000).
- It expects a JSONL dataset with fields: {id, image, prompt, response}.

If the model/processor class names change, the first thing to adjust is the
`AutoModelForVision2Seq` import and the processor usage.
"""

import argparse
import io
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from PIL import Image
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor | None
    labels: torch.Tensor


def build_chat_messages(prompt: str, response: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": response}]},
    ]


class Collator:
    def __init__(self, processor: Any, image_max_side: int, max_length: int):
        self.processor = processor
        self.image_max_side = image_max_side
        self.max_length = max_length

    def _coerce_image_source(self, value: Any) -> tuple[str, io.BytesIO | None]:
        """Return (path, bytes_buf) where exactly one is set.

        Supports:
        - str / Path: filesystem path (relative paths resolved against CWD)
        - dict: tries common keys like {path}, {image}, {file}, {bytes}
        - bytes-like: in-memory image bytes
        """
        if value is None:
            return ("", None)

        if isinstance(value, (bytes, bytearray, memoryview)):
            return ("", io.BytesIO(bytes(value)))

        if isinstance(value, str) and not value.strip():
            return ("", None)

        if isinstance(value, dict):
            if "bytes" in value and value["bytes"] is not None:
                b = value["bytes"]
                if isinstance(b, str):
                    # sometimes base64 strings show up; we don't decode here
                    return (b, None)
                if isinstance(b, (bytes, bytearray, memoryview)):
                    return ("", io.BytesIO(bytes(b)))
            for k in ("path", "image", "file", "filename"):
                if k in value and value[k]:
                    value = value[k]
                    break

        if isinstance(value, Path):
            p = value
        else:
            p = Path(str(value))

        # Path("") becomes "."; treat as missing.
        if str(p) in {"", "."}:
            return ("", None)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        return (str(p), None)

    def _load_image(self, image_value: Any) -> Image.Image:
        path, buf = self._coerce_image_source(image_value)

        # Always open from a file-like object to avoid PIL path-detection edge cases.
        if buf is not None:
            img = Image.open(buf)
            img.load()
        else:
            if not path:
                raise FileNotFoundError(
                    "Empty image path in dataset record (did you include missing-image docs?)"
                )
            with open(path, "rb") as f:
                img = Image.open(f)
                img.load()

        img = img.convert("RGB")
        # Resize by long-side while preserving aspect ratio
        w, h = img.size
        m = max(w, h)
        if self.image_max_side and m > self.image_max_side:
            scale = self.image_max_side / float(m)
            nw, nh = int(w * scale), int(h * scale)
            img = img.resize((nw, nh))
        return img

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images = [self._load_image(f.get("image")) for f in features]
        prompts = [str(f["prompt"]) for f in features]
        responses = [str(f["response"]) for f in features]

        # Build chat text using the model's chat template when available.
        texts: list[str] = []
        for p, r in zip(prompts, responses, strict=True):
            msgs = build_chat_messages(p, r)
            if hasattr(self.processor, "apply_chat_template"):
                text = self.processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
            else:
                # fallback: concatenate
                text = f"USER: {p}\nASSISTANT: {r}"
            texts.append(text)

        enc = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")
        pixel_values = enc.get("pixel_values")
        image_grid_thw = enc.get("image_grid_thw")
        image_attention_mask = enc.get("image_attention_mask")

        # Labels: standard causal LM labels; mask padding tokens.
        labels = input_ids.clone()
        pad_id = getattr(self.processor, "tokenizer", None)
        pad_token_id = pad_id.pad_token_id if pad_id is not None else None
        if pad_token_id is not None:
            labels[input_ids == pad_token_id] = -100

        batch: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if pixel_values is not None:
            batch["pixel_values"] = pixel_values
        # Qwen3-VL expects grid metadata for positional embeddings.
        if image_grid_thw is not None:
            batch["image_grid_thw"] = image_grid_thw
        if image_attention_mask is not None:
            batch["image_attention_mask"] = image_attention_mask
        return batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--train", required=True, help="path to train jsonl")
    ap.add_argument("--val", default="", help="path to val jsonl")
    ap.add_argument("--out", default="outputs/qwen3vl-8b-qlora")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--image-max-side", type=int, default=1536)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="PyTorch dataloader workers for image preprocessing (try 4).",
    )
    ap.add_argument(
        "--no-grad-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (faster if VRAM allows).",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    dataset = load_dataset(
        "json",
        data_files={
            "train": args.train,
            **({"validation": args.val} if args.val else {}),
        },
    )

    # Drop records with missing/invalid images (common when train.jsonl was built without --skip-missing-images).
    # We keep this in-script so training can proceed without rebuilding the dataset.
    def _has_valid_image(ex: dict[str, Any]) -> bool:
        v = ex.get("image")
        if v is None:
            return False
        if isinstance(v, str) and not v.strip():
            return False
        try:
            p = Path(str(v))
        except Exception:
            return False
        if str(p) in {"", "."}:
            return False
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        return p.exists() and p.is_file()

    before_train = len(dataset["train"])
    dataset["train"] = dataset["train"].filter(_has_valid_image)
    after_train = len(dataset["train"])
    if after_train != before_train:
        print(f"Filtered train records with missing images: {before_train} -> {after_train}")

    if "validation" in dataset:
        before_val = len(dataset["validation"])
        dataset["validation"] = dataset["validation"].filter(_has_valid_image)
        after_val = len(dataset["validation"])
        if after_val != before_val:
            print(f"Filtered val records with missing images: {before_val} -> {after_val}")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    # QLoRA: load 4-bit base
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        load_in_4bit=True,
    )

    # PEFT helper often enables gradient checkpointing by default (saves VRAM, costs speed).
    # On 48GB GPUs you may want to disable it for throughput.
    prepare_sig = set(inspect.signature(prepare_model_for_kbit_training).parameters)
    prepare_kwargs: dict[str, Any] = {}
    if "use_gradient_checkpointing" in prepare_sig:
        prepare_kwargs["use_gradient_checkpointing"] = (not args.no_grad_checkpointing)
    model = prepare_model_for_kbit_training(model, **prepare_kwargs)
    if args.no_grad_checkpointing and hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            # Common projection names; adjust if needed for this model release
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
    )

    model = get_peft_model(model, lora)

    collator = Collator(
        processor=processor, image_max_side=args.image_max_side, max_length=args.max_len
    )

    # Transformers API compat: newer versions renamed `evaluation_strategy` -> `eval_strategy`.
    targs_kwargs: dict[str, Any] = {
        "output_dir": args.out,
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.batch,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.grad_accum,
        "fp16": True,
        "dataloader_num_workers": args.num_workers,
        "dataloader_pin_memory": True,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "evaluation_strategy": "steps" if args.val else "no",
        "eval_steps": args.save_steps if args.val else None,
        "report_to": "none",
        "remove_unused_columns": False,
    }

    sig_params = set(inspect.signature(TrainingArguments.__init__).parameters)
    if "evaluation_strategy" not in sig_params and "eval_strategy" in sig_params:
        targs_kwargs["eval_strategy"] = targs_kwargs.pop("evaluation_strategy")

    # Drop any kwargs not supported by the installed Transformers version.
    targs_kwargs = {k: v for k, v in targs_kwargs.items() if k in sig_params}

    targs = TrainingArguments(**targs_kwargs)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.out)

    # Save a minimal adapter config artifact for serving
    (Path(args.out) / "run_info.json").write_text(
        json.dumps(
            {
                "base_model": args.model,
                "method": "qlora",
                "image_max_side": args.image_max_side,
                "max_len": args.max_len,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

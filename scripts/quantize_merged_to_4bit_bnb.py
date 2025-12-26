import argparse
import os

import torch
from transformers import (AutoModelForVision2Seq, AutoProcessor,
                          BitsAndBytesConfig)


def _parse_compute_dtype(value: str) -> torch.dtype:
    v = value.strip().lower()
    if v in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if v in {"fp16", "float16"}:
        return torch.float16
    raise argparse.ArgumentTypeError(
        "--compute-dtype must be one of: bf16, fp16")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export a merged HF checkpoint as a BitsAndBytes 4-bit (NF4/FP4) "
            "checkpoint that vLLM can load."
        ))
    parser.add_argument(
        "--in",
        dest="input_dir",
        required=True,
        help="Path to the merged HF checkpoint directory (from merge_adapter_into_base.py).",
    )
    parser.add_argument(
        "--out",
        dest="output_dir",
        required=True,
        help="Output directory for the 4-bit BitsAndBytes checkpoint.",
    )
    parser.add_argument(
        "--quant-type",
        choices=["nf4", "fp4"],
        default="nf4",
        help="4-bit quantization type (default: nf4).",
    )
    parser.add_argument(
        "--compute-dtype",
        type=_parse_compute_dtype,
        default=torch.bfloat16,
        help="Compute dtype for 4-bit layers (default: bf16).",
    )
    parser.add_argument(
        "--max-shard-size",
        default="2GB",
        help="Max shard size for saved safetensors (default: 2GB).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading model/processor.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=args.quant_type,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=args.compute_dtype,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        args.input_dir,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=args.trust_remote_code,
    )
    processor = AutoProcessor.from_pretrained(
        args.input_dir,
        trust_remote_code=args.trust_remote_code,
    )

    model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    processor.save_pretrained(args.output_dir)

    print("Saved BitsAndBytes 4-bit checkpoint to:", args.output_dir)
    print("vLLM: load it directly (pre-quantized BitsAndBytes).")


if __name__ == "__main__":
    main()

"""Deprecated.

This repo targets vLLM deployment; Optimum-Quanto exports are not directly
vLLM-loadable.

Use `scripts/quantize_merged_to_4bit_bnb.py` instead.
"""

raise SystemExit(
    "Optimum-Quanto export removed (not vLLM-friendly).\n"
    "Use: python scripts\\quantize_merged_to_4bit_bnb.py --in <merged_dir> --out <bnb4_dir>"
)

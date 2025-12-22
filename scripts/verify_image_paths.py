import argparse
import json
from pathlib import Path
from typing import Any


def get_image_path(rec: dict[str, Any]) -> str:
    msgs = rec.get("messages")
    if not isinstance(msgs, list):
        return ""
    for m in msgs:
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                return str(item.get("image") or "")
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/train.jsonl")
    ap.add_argument("--out-missing", default="docs/missing_image_files.txt")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    cwd = Path.cwd().resolve()

    missing: list[str] = []
    total = 0

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            rec = json.loads(line)
            img = get_image_path(rec)
            if not img:
                missing.append(f"<no-image-field>\t{rec.get('id','')}")
                continue
            p = Path(img)
            if not p.is_absolute():
                p = (cwd / p).resolve()
            if not p.exists():
                missing.append(f"{img}\t{rec.get('id','')}")

    out_path = Path(args.out_missing)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")

    print(f"checked: {total}")
    print(f"missing files: {len(missing)} -> {out_path}")


if __name__ == "__main__":
    main()

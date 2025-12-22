import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def get_image_path(rec: dict[str, Any]) -> str:
    try:
        msgs = rec.get("messages", [])
        for m in msgs:
            if m.get("role") == "user":
                for item in m.get("content", []):
                    if item.get("type") == "image":
                        return str(item.get("image") or "")
    except Exception:
        return ""
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/train.jsonl")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    n = 0
    missing_image_field = 0
    image_ext = Counter()
    doc_types = Counter()
    containers_per_doc = []

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            rec = json.loads(line)
            img = get_image_path(rec)
            if not img:
                missing_image_field += 1
            else:
                suffix = Path(img).suffix.lower() or "(none)"
                image_ext[suffix] += 1

            meta = rec.get("meta") or {}
            if isinstance(meta, dict):
                dt = meta.get("doc_type")
                if dt:
                    doc_types[str(dt)] += 1

            # containers count from assistant json
            try:
                assistant_text = rec["messages"][-1]["content"][0]["text"]
                obj = json.loads(assistant_text)
                cd = obj.get("container_details")
                if isinstance(cd, list):
                    containers_per_doc.append(len(cd))
            except Exception:
                pass

    print(f"records: {n}")
    print(f"missing image field: {missing_image_field}")
    if image_ext:
        print("image extensions:")
        for k, v in image_ext.most_common():
            print(f"  {k}: {v}")

    if containers_per_doc:
        containers_per_doc.sort()
        print(
            "containers/doc: "
            f"min={containers_per_doc[0]} "
            f"p50={containers_per_doc[len(containers_per_doc)//2]} "
            f"max={containers_per_doc[-1]}"
        )


if __name__ == "__main__":
    main()

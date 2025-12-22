import argparse
import json
from pathlib import Path
from typing import Any


def _get_prompt_and_image(rec: dict[str, Any]) -> tuple[str, str]:
    """Extract the user prompt text and image path from our chat-style record."""
    prompt = ""
    image = ""
    for msg in rec.get("messages", []):
        if msg.get("role") != "user":
            continue
        for item in msg.get("content", []):
            if item.get("type") == "text" and not prompt:
                prompt = str(item.get("text") or "")
            elif item.get("type") == "image" and not image:
                image = str(item.get("image") or "")
    return prompt, image


def _get_response(rec: dict[str, Any]) -> str:
    """Extract assistant JSON string from our chat-style record."""
    msgs = rec.get("messages", [])
    if not msgs:
        return ""
    last = msgs[-1]
    if last.get("role") != "assistant":
        return ""
    content = last.get("content") or []
    if not content:
        return ""
    first = content[0]
    if first.get("type") != "text":
        return ""
    return str(first.get("text") or "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument(
        "--format",
        choices=["simple", "messages"],
        default="simple",
        help=(
            "simple: {id,image,prompt,response}; messages: {id,messages}"
        ),
    )
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = str(rec.get("id") or "")
            if args.format == "messages":
                obj = {"id": rid, "messages": rec.get("messages", [])}
            else:
                prompt, image = _get_prompt_and_image(rec)
                response = _get_response(rec)
                obj = {"id": rid, "image": image, "prompt": prompt, "response": response}
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} record(s) -> {out_path}")


if __name__ == "__main__":
    main()

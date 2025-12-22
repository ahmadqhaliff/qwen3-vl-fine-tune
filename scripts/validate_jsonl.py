import json
import sys
from pathlib import Path
from typing import Any


def fail(msg: str) -> None:
    raise SystemExit(msg)


def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and len(x.strip()) > 0


def validate_record(obj: dict[str, Any], line_no: int) -> list[str]:
    errors: list[str] = []

    if not isinstance(obj, dict):
        return [f"line {line_no}: record is not an object"]

    if not _is_nonempty_str(obj.get("id")):
        errors.append(f"line {line_no}: missing/invalid 'id'")

    messages = obj.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        errors.append(f"line {line_no}: 'messages' must be an array with >=2 items")
        return errors

    for mi, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"line {line_no}: messages[{mi}] is not an object")
            continue

        role = msg.get("role")
        if role not in ("system", "user", "assistant"):
            errors.append(f"line {line_no}: messages[{mi}].role invalid: {role!r}")

        content = msg.get("content")
        if not isinstance(content, list) or len(content) < 1:
            errors.append(f"line {line_no}: messages[{mi}].content must be a non-empty array")
            continue

        for ci, item in enumerate(content):
            if not isinstance(item, dict):
                errors.append(f"line {line_no}: messages[{mi}].content[{ci}] is not an object")
                continue
            t = item.get("type")
            if t == "text":
                if not _is_nonempty_str(item.get("text")):
                    errors.append(f"line {line_no}: messages[{mi}].content[{ci}] missing/invalid text")
            elif t == "image":
                if not _is_nonempty_str(item.get("image")):
                    errors.append(f"line {line_no}: messages[{mi}].content[{ci}] missing/invalid image path")
            else:
                errors.append(f"line {line_no}: messages[{mi}].content[{ci}].type invalid: {t!r}")

    # Strongly recommended for SFT: last message should be assistant text-only JSON
    last = messages[-1]
    if isinstance(last, dict) and last.get("role") == "assistant":
        content = last.get("content")
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and first.get("type") == "text":
                txt = first.get("text")
                if isinstance(txt, str):
                    try:
                        json.loads(txt)
                    except Exception:
                        errors.append(
                            f"line {line_no}: assistant content[0].text is not valid JSON; "
                            "for extraction, prefer strict JSON"
                        )

    return errors


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        fail("Usage: python scripts\\validate_jsonl.py <path-to-jsonl>")

    path = Path(argv[1])
    if not path.exists():
        fail(f"File not found: {path}")

    total = 0
    all_errors: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception as e:
                all_errors.append(f"line {line_no}: invalid JSON: {e}")
                continue
            all_errors.extend(validate_record(obj, line_no))

    if all_errors:
        print("FAILED")
        for e in all_errors:
            print("-", e)
        raise SystemExit(2)

    print(f"OK: {total} record(s) validated")


if __name__ == "__main__":
    main(sys.argv)

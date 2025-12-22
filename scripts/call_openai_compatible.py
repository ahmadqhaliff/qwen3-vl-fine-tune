import base64
import json
import sys
from pathlib import Path
from typing import Any

import urllib.request


def b64_image(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("ascii")


def main(argv: list[str]) -> None:
    if len(argv) < 4:
        raise SystemExit(
            "Usage: python scripts\\call_openai_compatible.py <base_url> <model> <image_path> [max_tokens]"
        )

    base_url = argv[1].rstrip("/")
    model = argv[2]
    image_path = Path(argv[3])
    max_tokens = int(argv[4]) if len(argv) >= 5 else 256

    # OpenAI-compatible chat.completions payload with an image.
    # Note: exact multimodal payload shape can vary by server/version.
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{b64_image(image_path)}",
                    },
                    {
                        "type": "input_text",
                        "text": "Extract fields and return ONLY JSON.",
                    },
                ],
            }
        ],
    }

    req = urllib.request.Request(
        url=f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        body = resp.read().decode("utf-8")
        print(body)


if __name__ == "__main__":
    main(sys.argv)

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    examples_dir = root / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    # This image is a placeholder path (no file needed for JSONL generation).
    # When you provide PDFs, we'll render pages and populate real paths.
    example = {
        "id": "sample-invoice-0001_p1",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a document information extraction engine. "
                            "Return ONLY valid JSON that conforms to the requested schema."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "data/images/sample-invoice-0001_page_0001.png"},
                    {
                        "type": "text",
                        "text": (
                            "Extract fields from this invoice page. "
                            "Output JSON with keys: invoice_number, invoice_date, seller_name, "
                            "buyer_name, currency, total_amount. "
                            "Use null if missing."
                        ),
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "invoice_number": "INV-000123",
                                "invoice_date": "2025-10-03",
                                "seller_name": "ACME Supplies Sdn Bhd",
                                "buyer_name": "Example Buyer Sdn Bhd",
                                "currency": "MYR",
                                "total_amount": 1234.56,
                            },
                            ensure_ascii=False,
                        ),
                    }
                ],
            },
        ],
        "meta": {
            "doc_type": "invoice",
            "page": 1,
            "source": "placeholder",
        },
    }

    out_path = examples_dir / "example.jsonl"
    out_path.write_text(json.dumps(example, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

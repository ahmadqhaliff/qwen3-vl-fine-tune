import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def norm_key(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def prefer_image(paths: list[Path]) -> Path:
    def score(p: Path) -> tuple[int, int, str]:
        name = p.name.lower()
        # lower is better
        kind = 3
        if "single_page" in name:
            kind = 0
        elif "combined_grid" in name:
            kind = 1
        elif re.search(r"_\d+\.(jpg|png)$", name):
            kind = 2

        # if numbered pages, pick lowest page number
        page = 10**9
        m = re.search(r"_(\d+)\.(jpg|png)$", name)
        if m:
            page = int(m.group(1))

        # stable tie-break
        return (kind, page, name)

    return sorted(paths, key=score)[0]


def normalize_port_of_discharge(value: str) -> str:
    v = (value or "").strip()
    u = v.upper()
    if "KLANG" in u:
        return "Port Klang"
    if "PASIR GUDANG" in u:
        return "Pasir Gudang"
    if "TANJUNG PELEPAS" in u or re.search(r"\bPTP\b", u):
        return "Port Tanjung Pelepas"
    if "PENANG" in u:
        return "Penang"
    return v


def to_str_or_empty(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="docs/results.json")
    ap.add_argument(
        "--images-dir",
        action="append",
        default=[],
        help="directory containing extracted images (repeatable)",
    )
    ap.add_argument("--prompt", default="prompts/bl_extraction_prompt.txt")
    ap.add_argument("--out", default="data/train.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="limit number of documents (0 = all)")
    ap.add_argument("--skip-missing-images", action="store_true", help="skip docs if no matching image found")
    ap.add_argument(
        "--missing-report",
        default="",
        help="write missing source filenames (no matching image) to this path",
    )
    args = ap.parse_args()

    results_path = Path(args.results)
    images_dirs = [Path(p) for p in (args.images_dir or [])]
    if not images_dirs:
        images_dirs = [Path("data/raw/combined")]
    prompt_path = Path(args.prompt)
    out_path = Path(args.out)

    if not results_path.exists():
        raise SystemExit(f"Results not found: {results_path}")
    for images_dir in images_dirs:
        if not images_dir.exists():
            raise SystemExit(
                f"Images dir not found: {images_dir} (run scripts\\extract_combined_zip.py first)"
            )
    if not prompt_path.exists():
        raise SystemExit(f"Prompt not found: {prompt_path}")

    prompt_text = prompt_path.read_text(encoding="utf-8")

    # Index images by normalized stem for fast matching.
    cwd = Path.cwd().resolve()

    images: list[Path] = []
    for d in images_dirs:
        for ext in ("*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"):
            images.extend(d.rglob(ext))

    norm_to_paths: dict[str, list[Path]] = defaultdict(list)
    for p in images:
        norm_to_paths[norm_key(p.stem)].append(p)

    # Load tabular extraction results (one container per row).
    rows = json.loads(results_path.read_text(encoding="utf-8"))

    by_file: dict[str, dict[str, Any]] = {}

    for r in rows:
        fn = r.get("Filename")
        if not fn:
            continue

        entry = by_file.get(fn)
        if entry is None:
            entry = {
                "Filename": fn,
                "consignee_name": r.get("consignee_name", ""),
                "bl_number": r.get("bl_number", ""),
                "port_of_loading": r.get("port_of_loading", ""),
                "port_of_discharge": r.get("port_of_discharge", ""),
                "vessel_name": r.get("vessel_name", ""),
                "detention_free_days": r.get("detention_free_days"),
                "demurrage_free_days": r.get("demurrage_free_days"),
                "combined_free_days": r.get("combined_free_days"),
                "container_details": [],
                "_seen_containers": set(),
            }
            by_file[fn] = entry

        cnum = (r.get("Container_Number") or "").strip()
        if cnum and cnum not in entry["_seen_containers"]:
            entry["_seen_containers"].add(cnum)
            entry["container_details"].append(
                {
                    "container_number": cnum,
                    "container_size": to_str_or_empty(r.get("Container_Size")),
                    "container_type": to_str_or_empty(r.get("Container_Type")),
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    missing: list[str] = []

    with out_path.open("w", encoding="utf-8") as out:
        for fn, entry in by_file.items():
            if args.limit and written >= args.limit:
                break

            base = fn[:-4] if fn.lower().endswith(".pdf") else fn
            nbase = norm_key(base)

            # Find candidate image paths whose normalized stem starts with the normalized base.
            candidates: list[Path] = []
            for k, vs in norm_to_paths.items():
                if k.startswith(nbase):
                    candidates.extend(vs)

            if not candidates:
                missing.append(fn)
                if args.skip_missing_images:
                    skipped += 1
                    continue
                image_rel = ""
            else:
                chosen = prefer_image(candidates)
                try:
                    image_rel = chosen.resolve().relative_to(cwd).as_posix()
                except Exception:
                    image_rel = chosen.as_posix()

            container_details = entry["container_details"]
            total_expected = len(container_details) if container_details else None

            assistant_obj = {
                "consignee_name": to_str_or_empty(entry.get("consignee_name")),
                "bl_number": to_str_or_empty(entry.get("bl_number")),
                "port_of_loading": to_str_or_empty(entry.get("port_of_loading")),
                "port_of_discharge": normalize_port_of_discharge(to_str_or_empty(entry.get("port_of_discharge"))),
                "vessel_name": to_str_or_empty(entry.get("vessel_name")),
                "detention_free_days": to_str_or_empty(entry.get("detention_free_days")),
                "demurrage_free_days": to_str_or_empty(entry.get("demurrage_free_days")),
                "combined_free_days": to_str_or_empty(entry.get("combined_free_days")),
                "total_expected_containers": total_expected,
                "container_details": container_details,
            }

            record = {
                "id": norm_key(fn),
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "Return ONLY valid JSON. No extra text.",
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image_rel if image_rel else "",
                            },
                            {"type": "text", "text": prompt_text},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(assistant_obj, ensure_ascii=False),
                            }
                        ],
                    },
                ],
                "meta": {
                    "filename": fn,
                    "image_rel": image_rel,
                },
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} record(s) to {out_path}")
    if skipped:
        print(f"Skipped {skipped} doc(s) with missing images")

    if args.missing_report:
        report_path = Path(args.missing_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        # de-dupe while preserving stable order
        seen: set[str] = set()
        unique_missing: list[str] = []
        for fn in missing:
            if fn in seen:
                continue
            seen.add(fn)
            unique_missing.append(fn)
        report_path.write_text("\n".join(unique_missing) + ("\n" if unique_missing else ""), encoding="utf-8")
        print(f"Missing-image report: {len(unique_missing)} filename(s) -> {report_path}")


if __name__ == "__main__":
    main()

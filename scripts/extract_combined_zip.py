import argparse
import zipfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--zip",
        dest="zip_paths",
        action="append",
        default=[],
        help="path to a zip file (repeatable)",
    )
    ap.add_argument(
        "--out",
        dest="out_dir",
        default="",
        help=(
            "output directory. If omitted, each zip is extracted to data/raw/<zip_stem>/"
        ),
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    zip_paths = [Path(p) for p in (args.zip_paths or [])]
    if not zip_paths:
        zip_paths = [Path("data/combined.zip")]

    total_extracted = 0
    total_skipped = 0

    for zip_path in zip_paths:
        if not zip_path.exists():
            raise SystemExit(f"Zip not found: {zip_path}")

        out_dir = Path(args.out_dir) if args.out_dir else (Path("data/raw") / zip_path.stem)
        out_dir.mkdir(parents=True, exist_ok=True)

        extracted = 0
        skipped = 0
        with zipfile.ZipFile(zip_path) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue

                target = out_dir / info.filename
                target.parent.mkdir(parents=True, exist_ok=True)

                if target.exists() and not args.overwrite:
                    skipped += 1
                    continue

                with z.open(info) as src, target.open("wb") as dst:
                    dst.write(src.read())
                extracted += 1

        total_extracted += extracted
        total_skipped += skipped
        print(f"Extracted {extracted} file(s) from {zip_path} to {out_dir}")
        if skipped:
            print(f"Skipped {skipped} existing file(s) for {zip_path}")

    if len(zip_paths) > 1:
        print(f"Total extracted: {total_extracted} file(s)")
        if total_skipped:
            print(f"Total skipped: {total_skipped} existing file(s)")


if __name__ == "__main__":
    main()

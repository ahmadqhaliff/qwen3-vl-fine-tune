import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/train.jsonl")
    ap.add_argument("--out-dir", default="data/splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.9)
    ap.add_argument("--val", type=float, default=0.05)
    ap.add_argument("--test", type=float, default=0.05)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)

    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    ratios = [args.train, args.val, args.test]
    if any(r < 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise SystemExit("Ratios must be non-negative and sum to 1.0")

    records: list[str] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                # ensure valid json
                json.loads(line)
                records.append(line)

    rng = random.Random(args.seed)
    rng.shuffle(records)

    n = len(records)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    n_test = n - n_train - n_val

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.jsonl").write_text("".join(records[:n_train]), encoding="utf-8")
    (out_dir / "val.jsonl").write_text("".join(records[n_train : n_train + n_val]), encoding="utf-8")
    (out_dir / "test.jsonl").write_text("".join(records[n_train + n_val :]), encoding="utf-8")

    print(f"Wrote splits to {out_dir}")
    print(f"train: {n_train}  val: {n_val}  test: {n_test}  total: {n}")


if __name__ == "__main__":
    main()

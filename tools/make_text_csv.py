"""Convert a fixed-caption evaluation TSV to the evaluator's name,caption CSV."""
import argparse
import csv
from pathlib import Path


def convert(manifest: Path, output: Path) -> int:
    if output.exists():
        raise FileExistsError(output)
    with manifest.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not {"id", "caption"}.issubset(reader.fieldnames or []):
            raise ValueError("manifest needs id and caption columns")
        rows = list(reader)
    ids = [row["id"] for row in rows]
    if not rows or len(set(ids)) != len(ids) or any(not value for value in ids):
        raise ValueError("manifest IDs must be nonempty and unique")
    if any(not row["caption"] for row in rows):
        raise ValueError("captions must be nonempty")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["name", "caption"])
        writer.writeheader()
        writer.writerows({"name": row["id"], "caption": row["caption"]} for row in rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"wrote {convert(args.manifest, args.output)} rows to {args.output}")


if __name__ == "__main__":
    main()

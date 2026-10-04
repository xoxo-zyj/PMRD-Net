"""Check the size and SHA-256 of every distributed inference weight."""
import argparse
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "checkpoints/manifest.json").read_text(encoding="utf-8"))
    failed = []
    for entry in manifest["weights"]:
        path = (root / entry["file"]).resolve()
        if root not in path.parents:
            raise ValueError(f"Invalid weight path: {entry['file']}")
        if not path.is_file():
            failed.append(entry["file"] + " (missing)")
        elif path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            failed.append(entry["file"] + " (checksum mismatch)")
        else:
            print("OK:", entry["file"])
    if failed:
        print("\nMerge MSDSPDD_Model_Weights.zip into this repository, then retry.")
        for message in failed:
            print("FAIL:", message)
    return not failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    raise SystemExit(0 if verify(args.root) else 1)


if __name__ == "__main__":
    main()

"""Resolve portable manifests for local datasets without changing membership."""
import argparse
import csv
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATH_FIELDS = ("hazy_path", "gt_path", "group_id")


def portable_path(value):
    value = value.replace("\\", "/")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or PureWindowsPath(value).drive:
        raise ValueError(f"Expected a relative dataset path: {value!r}")
    if ".." in path.parts:
        raise ValueError(f"Parent traversal is not allowed: {value!r}")
    return path.as_posix()


def load_prefix_map(path):
    if path is None:
        return {}
    mapping = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError("Prefix mapping must be a JSON object.")
    return {portable_path(k): portable_path(v) for k, v in mapping.items()}


def resolve_path(value, data_root, mapping):
    relative = portable_path(value)
    for old in sorted(mapping, key=len, reverse=True):
        if relative == old or relative.startswith(old + "/"):
            relative = mapping[old] + relative[len(old):]
            break
    root = Path(data_root)
    if not root.is_absolute():
        root = root.resolve()
    return str(root / relative)


def read_manifest(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields or not set(PATH_FIELDS).issubset(fields):
            raise ValueError(f"Missing path columns: {path}")
        rows = list(reader)
    for row in rows:
        for field in PATH_FIELDS:
            portable_path(row[field])
    return fields, rows


def audit_disjoint(manifests):
    """GT groups, hazy files and GT files must stay disjoint across splits."""
    by_dataset = {}
    for relative, _, rows in manifests:
        dataset = relative.parts[0]
        previous = by_dataset.setdefault(dataset, {key: set() for key in PATH_FIELDS})
        for key in PATH_FIELDS:
            current = {row[key] for row in rows}
            overlap = previous[key] & current
            if overlap:
                raise ValueError(f"Cross-split overlap in {dataset}/{key}: {next(iter(overlap))}")
            previous[key].update(current)


def prepare(source_root, output_root, data_root, mapping=None, check_files=False):
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    if output_root == source_root or source_root in output_root.parents or output_root in source_root.parents:
        raise ValueError("Output and source manifest directories must be separate.")
    files = sorted(source_root.glob("*/split_manifest_*.csv"))
    if not files:
        raise ValueError(f"No split manifests found under {source_root}")
    originals = [(path.relative_to(source_root), *read_manifest(path)) for path in files]
    audit_disjoint(originals)
    mapping = mapping or {}
    data_root = Path(data_root).resolve()
    resolved_paths = {}
    converted = []
    missing = []
    for relative, fields, rows in originals:
        mapped_rows = []
        for row in rows:
            mapped = dict(row)
            for key in PATH_FIELDS:
                if row[key] not in resolved_paths:
                    resolved_paths[row[key]] = resolve_path(row[key], data_root, mapping)
                mapped[key] = resolved_paths[row[key]]
            if check_files:
                for key in ("hazy_path", "gt_path"):
                    if not Path(mapped[key]).is_file():
                        missing.append(mapped[key])
            mapped_rows.append(mapped)
        converted.append((relative, fields, mapped_rows))
    # Mapping must not merge originally independent groups or files.
    audit_disjoint(converted)
    if missing:
        examples = "\n".join(dict.fromkeys(missing[:5]))
        raise FileNotFoundError(f"Missing {len(missing)} image references (examples):\n{examples}")
    for relative, fields, rows in converted:
        target = output_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(target)
        print(f"{relative.as_posix()}: {len(rows)} pairs")
    print(f"Written {len(converted)} manifests to {output_root}; membership unchanged.")
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "data/splits")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data/local_splits")
    parser.add_argument("--prefix-map", type=Path)
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()
    try:
        prepare(args.source_root, args.output_root, args.data_root,
                load_prefix_map(args.prefix_map), args.check_files)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Split preparation failed: {error}\n")


if __name__ == "__main__":
    main()

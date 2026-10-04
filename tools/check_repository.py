"""Framework-free source, split and CLI checks for packaging and GitHub CI."""
import argparse
import ast
import contextlib
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from tools.prepare_splits import read_manifest, audit_disjoint


def parse_cli(filename, arguments):
    """Run only stdlib argument parsing, without importing model dependencies."""
    tree = ast.parse((PROJECT_ROOT / filename).read_text(encoding="utf-8-sig"))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and (t.id == "PROJECT_ROOT" or t.id == "TASK_ORDER" or t.id.startswith("DEFAULT_")) for t in node.targets):
                nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "parse_args":
            nodes.append(node)
    namespace = {"__file__": str(PROJECT_ROOT / filename), "Path": Path,
                 "argparse": argparse, "sys": sys}
    compiled = compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec")
    original = sys.argv
    try:
        sys.argv = [filename] + list(arguments)
        exec(compiled, namespace)
        return namespace["parse_args"]()
    finally:
        sys.argv = original


def main():
    python_files = [p for p in PROJECT_ROOT.rglob("*.py") if not any(part in {"vendor", ".venv", "venv"} for part in p.relative_to(PROJECT_ROOT).parts)]
    for path in python_files:
        compile(path.read_bytes(), str(path), "exec")
    preservation = json.loads((PROJECT_ROOT / "reports/source_preservation.json").read_text(encoding="utf-8"))
    for record in preservation["unchanged_source_files"]:
        assert hashlib.sha256((PROJECT_ROOT / record["path"]).read_bytes()).hexdigest() == record["sha256"], record["path"]
    summaries = json.loads((PROJECT_ROOT / "reports/split_manifest_summary.json").read_text(encoding="utf-8"))
    manifests = []
    total = 0
    for record in summaries:
        relative = Path(record["dataset"]) / record["file"]
        path = PROJECT_ROOT / "data/splits" / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["published_sha256"], str(relative)
        fields, rows = read_manifest(path)
        assert len(rows) == record["pairs"], str(relative)
        manifests.append((relative, fields, rows))
        total += len(rows)
    audit_disjoint(manifests)
    training = parse_cli("finetune.py", [])
    assert training.keep_old_step_outputs
    assert not parse_cli("finetune.py", ["--purge_old_step_outputs"]).keep_old_step_outputs
    assert Path(training.base_ckpt) == PROJECT_ROOT / "checkpoints/general/general_best.pth"
    evaluation = parse_cli("evaluate.py", [])
    assert Path(evaluation.split_root) == PROJECT_ROOT / "data/local_splits"
    for filename in ("finetune.py", "evaluate.py"):
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                parse_cli(filename, ["--help"])
            except SystemExit as error:
                assert error.code == 0
    protocols = json.loads((PROJECT_ROOT / "reports/protocol_names.json").read_text(encoding="utf-8"))
    assert "tile512_stride256" in protocols
    assert not list(PROJECT_ROOT.rglob("*.pth")) and not list(PROJECT_ROOT.rglob("*.pt")), "Keep binary weights outside the code upload."
    print(f"PASS: {len(python_files)} Python files; {len(preservation['unchanged_source_files'])} unchanged sources; {len(manifests)} split manifests / {total} pairs; CLI defaults and help.")


if __name__ == "__main__":
    main()

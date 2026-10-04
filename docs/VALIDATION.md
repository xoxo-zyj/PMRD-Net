# Packaging validation

Checks performed on the preparation host (Python 3.12, Windows):

- Compile all 30 Python files without executing model imports.
- Compare SHA-256 for 21 verbatim source files with the extracted original.
- Verify all 15 portable manifests and their 79,203 pair records against the
  packaging manifests; audit hazy/GT/group separation across splits.
- Execute only the stdlib CLI parser portions of the two active drivers;
  check help, portable defaults and explicit opt-in checkpoint cleanup.
- Run five regression tests for absolute/traversal path rejection, longest-prefix
  remapping, missing-image handling, record preservation and split leakage checks.
- Generate local paths for all archived split records in a separate scratch
  directory without changing the public split files.
- Verify all 18 original weight binaries are accounted for across the two binary
  assets; verify source-to-ZIP SHA-256 and ZIP CRC integrity during packaging.

The distributed code CI runs the framework-free checks on Python 3.10 and 3.12.
Only Python 3.12 was executed locally; the GitHub Actions runs will occur after
uploading. These checks validate packaging and path handling, not dehazing quality.

## Runtime limitation

The host has no Torch, torchvision, CUDA or dataset images. Full-model GPU
forward/backward execution, checkpoint loading into the model, training,
dataset-image validation and metric reproduction were not performed.
The environment preflight correctly reports the missing dependencies on this host.
The EfficientViT baseline and custom selective-scan build need verification on
the intended CUDA machine using [INSTALL.md](INSTALL.md).

## Repeat code checks

From a fresh code checkout:

```bash
python tools/check_repository.py
python -m unittest discover -s tests -v
```

Weights have a separate `tools/verify_weights.py` check after merging the model
asset. Asset-level checksums are supplied next to the release ZIPs.

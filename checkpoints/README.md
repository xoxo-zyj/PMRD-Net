# Model checkpoints

Extract `MSDSPDD_Model_Weights.zip` next to the repository folder so its
`MSDSPDD/` directory merges with this repository. Verify with:

```bash
python tools/verify_weights.py
```

The inference asset contains one EfficientViT-B1 initialization file, the
best general checkpoint, and five dataset-specific model-only checkpoints.
Exact filenames, original archive paths, sizes and SHA-256 values are in
[manifest.json](manifest.json). Binary weights are ignored by Git.

Full best/latest optimizer checkpoints and original general pair reports
are retained in the separate `MSDSPDD_Training_Archive.zip` asset.

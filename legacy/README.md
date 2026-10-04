# Archived entrypoints

`original_entrypoints/` retains every root-level Python entrypoint from the
source archive verbatim, including both older fine-tuning versions and the
original checked epoch trainer/evaluator. They contain original server paths
and are kept for comparison, not as the recommended launch commands.

Use root-level `finetune.py`, `evaluate.py` and `smoke_test.py` for the
published interface. `backups/` retains the original EfficientViT init backup.

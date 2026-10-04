# Running the archived implementation

The active drivers import the modular implementation in `future_process/`.
`msdspdd_complete_general.py` is also retained as the original single-file
model/loss implementation; equivalence between those two implementations has
not been established by this packaging work.

## Fine-tuning

After installing the environment, merging weights and preparing local splits:

```bash
python finetune.py --dataset ihaze --check_only
python finetune.py --dataset ihaze --amp
```

Valid task arguments: `ihaze`, `ohaze`, `nhhaze`, `sots_indoor`, `sots_outdoor`,
or `all`. The checkpoint default is `checkpoints/general/general_best.pth`;
each task starts from the archived general checkpoint. Outputs go to
`runs/finetune/<task>/`. The default seed is 42. Original training behavior,
stage unfreezing, early stopping and schedules are preserved. Dataset defaults
allow up to 10,000 epochs with early stopping; set `--max_epochs` explicitly for
a short verification run. `--check_only` still requires the ML dependencies.

```bash
python finetune.py --dataset ihaze --max_epochs 1 --batch_size 1 --num_workers 0 --amp
```

This short run changes training duration and is a runtime check, not a reproduction
of the released weights. `--force_restart`, `--no_skip_finished` and other original
options are listed by `python finetune.py --help`.

Old step outputs are preserved by default. `--purge_old_step_outputs` explicitly
enables the original cleanup against `--old_step_output_root`; the legacy scripts
retain their original behavior and should only be used for archival comparison.

The archive lacks an independent general pretraining driver. Its general best
and latest checkpoints and pairing reports are provided, but those are not a
replacement for the absent training entrypoint.

## Evaluation

```bash
python evaluate.py --datasets I-HAZE --protocols tile512_stride256 --save_images
python evaluate.py --datasets I-HAZE --protocols tile512_stride256 --finetune_root runs/finetune
```

The first command uses released fine-tuned weights, the second uses your local
fine-tuning outputs. The tile protocol here is an example, not a claim about which
protocol produced the paper's reported results. HazeRD uses the general best
checkpoint. Results default to `results/evaluation/`.

Dataset names are case-sensitive: `I-HAZE`, `O-HAZE`, `NH-HAZE`, `HazeRD`,
`SOTS-indoor`, `SOTS-outdoor`. Use comma-separated names or `all`.

All eight original protocol names are:

```text
tile1024_stride512
tile768_stride384
tile640_stride320
tile512_stride256
tile384_stride192
tile320_stride160
tile256_stride128
resize256
```

`resize256` resizes using `--resize_size` (default 256). The tiling protocols use
the named tile/stride. NIQE/BRISQUE use `--iqa_resize 512` by default; zero keeps
the protocol output size. The default metric shave is zero and evaluation is
FP32 unless `--amp` is explicitly requested.

The archived evaluator defaults to `--datasets all --protocols all` and creates
a best-PSNR-per-dataset summary after scanning protocols. Selecting a protocol on
test-set scores biases a formal benchmark comparison. Preserve the sweep as
exploratory analysis and select a formal reporting protocol independently.

## Record with experimental results

Save the repository revision, `python -m pip freeze`, Python/Torch/torchvision,
CUDA/driver/kernel build versions, GPU, dataset source and resolved split manifests,
checkpoint SHA-256, CLI arguments, metric configuration, pair/unmatched counts,
seeds and smoke-test output. No new PSNR/SSIM/NIQE/BRISQUE results were generated
during packaging.

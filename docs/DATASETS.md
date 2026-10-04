# Datasets and supplied splits

Dataset images are not included. Obtain each dataset from its official provider
under its applicable terms. The archive supplied filenames and training/split
records, not the images themselves.

## Preserved membership

| Task | Train pairs | Validation pairs | Reserved test pairs |
| --- | ---: | ---: | ---: |
| `ihaze` | 20 | 5 | 5 |
| `ohaze` | 35 | 5 | 5 |
| `nhhaze` | 45 | 5 | 5 |
| `sots_indoor` | 11,980 | 630 | 0 |
| `sots_outdoor` | 65,798 | 665 | 0 |

These counts describe the actual archived manifests. They are not asserted to
be official dataset-wide partitions. Repeated hazy versions of the same clean
image remain in the same split. The synthetic training manifests reference
ITS/OTS source directories; their task names do not mean they train on SOTS test
images. Synthetic evaluation reads the separate SOTS image folders.

`data/splits/` contains 15 manifests, preserving all 79,203 pairs, row order,
split labels, pairing annotations and GT grouping. Only the original
`/root/autodl-tmp/` path prefix was removed. Original absolute manifests remain
in `MSDSPDD_Training_Archive.zip` for provenance.

## Materialize local paths

If the original folder structure exists below your dataset root:

```text
datasets/
├── I-HAZY/hazy/ and I-HAZY/GT/
├── O-HAZE/hazy/ and O-HAZE/GT/
├── NH-HAZE/                         # names from the archived manifest
├── train/train/ITS_v2/
├── train/train/OTS_BETA/
├── remade_clear/ITS_v2/
└── remade_clear/OTS_BETA/
```

```bash
python tools/prepare_splits.py --data-root /path/to/datasets --check-files
```

If folders differ, copy and edit `configs/dataset_paths.example.json` so each
source prefix maps to the corresponding relative folder under your data root:

```bash
python tools/prepare_splits.py --data-root /path/to/datasets --prefix-map configs/dataset_paths.example.json --check-files
```

The example maps ITS/OTS images to `RESIDE/ITS/{hazy,clear}` and
`RESIDE/OTS_BETA/{hazy,clear}`. It changes directories only: original image
filenames must still match. For an `I-HAZE` directory, change the example mapping
from `"I-HAZY": "I-HAZY"` to `"I-HAZY": "I-HAZE"`.

The tool writes absolute paths to ignored `data/local_splits/`, checks
cross-split hazy/GT/group disjointness, and preserves split membership. With
`--check-files`, missing images cause failure before writing output manifests.
An empty synthetic reserved-test CSV is preserved and is expected.

## Evaluation image directories

I-HAZE/O-HAZE/NH-HAZE use the local `test_reserved` CSVs. SOTS and HazeRD are
paired by the archived evaluator's filename-matching logic:

```text
data/
├── HazeRD/hazy/ and HazeRD/GT/
└── SOTS/
    ├── indoor/hazy/ and indoor/clear/
    └── outdoor/hazy/ and outdoor/clear/
```

Override paths when images live elsewhere:

```bash
python evaluate.py --datasets SOTS-indoor --protocols tile512_stride256 --sots_root /path/to/SOTS
python evaluate.py --datasets HazeRD --protocols tile512_stride256 --hazerd_root /path/to/HazeRD
```

Inspect the logged pair/unmatched counts before reporting scores. The original
filename matcher can omit unmatched images or resolve ambiguous names by its
existing heuristic; this behavior was retained rather than reimplementing pairing.

# Reproduction environment — Iteration 2

Captured automatically at the start of `run_iteration2.py`. Use the same 
Python interpreter and package versions listed here to reproduce `Iteration2_Results.csv`.

## Interpreter

- Python: `3.12.10`
- Executable: `Slicer-5.12.3-linux-amd64/bin/python-real`
- Platform: `Linux`

## Required packages

| Package | Version |
|---|---|
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| pandas | 3.0.5 |
| nibabel | 5.4.2 |
| totalsegmentator | 2.14.0 |
| torch | 2.13.0+cu129 |

## GPU

- CUDA available: `True`
- Device: `NVIDIA GeForce RTX 4070 Ti`

## How to run

This script must run under Slicer's bundled Python (not system Python), 
which already ships nibabel/numpy/scipy/pandas. It is self-contained: it runs 
TotalSegmentator itself (downloading the free `total`/`total_mr` model weights 
on first use if not already cached) and writes its own `Segmentations/` folder 
— nothing needs to exist beforehand except TestData/.

```
Slicer-5.12.3-linux-amd64/bin/PythonSlicer run_iteration2.py
```

Expects, relative to the project root: `TestData/Images/`, `TestData/Labels/`, 
`TestData/Labels.txt`. Nothing else.

## Reproducibility note

`gt_organ_mean`, `gt_organ_std`, and `gt_cnr` are exactly reproducible on any 
re-run — they only depend on the ground-truth masks and raw image data, both 
static inputs. `it1_*`/`it2_*` columns depend on a fresh TotalSegmentator GPU 
inference pass each run, which is not perfectly bit-deterministic: expect 
agreement to 3-4 significant figures, not exact equality, across separate runs 
(even on the same machine). `it1_*` and `it2_*` within the SAME run are always 
computed from the identical segmentation, so it1_dice == it2_dice etc. exactly.

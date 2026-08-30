# Agent trajectory — Organ-Aware Windowing session

A chronological record of what was built, tested, discovered, and fixed in this session, with file references for traceability.

## 1. Built the Slicer module

Created `OrganAwareWindowing/OrganAwareWindowing.py`, a scripted 3D Slicer module:
- Directory picker → scans for `.nii`/`.nii.gz` files.
- Modality combobox (CT/MRI) → drives which free TotalSegmentator task runs (`total` for CT, `total_mr` for MRI — verified via the installed extension's own `requiresLicense` flags that these are the free tasks, not the licensed ones).
- Organ combobox, dynamically populated from the live TotalSegmentator class map when available, falling back to a curated static list otherwise.
- "Segment Organ" button → runs TotalSegmentator, isolates the chosen organ's segment.
- "Optimize Contrast" button → computes a percentile-based (1st–99th) window/level from the organ's own intensity distribution and applies it as the volume's display window/level.
- "Save Scene (.mrb)" button → persists the volume + segmentation + optimized display settings as a Slicer Data Bundle.

Verified end-to-end (module load, folder scan, volume load, synthetic-data self-test) by launching the actual Slicer binary headlessly, not just by inspection.

**Bug caught and fixed:** the CT vs. MRI organ list initially leaked non-organ structures ("clavicula", "kidney cyst") into the combobox because the exclusion-keyword filter didn't cover TotalSegmentator's actual label spelling. Found by dumping the live `class_map` and diffing against the filtered list; fixed by extending the keyword filter.

## 2. Baseline CNR / intensity metrics (ground truth)

`TestData/compute_cnr.py`: for each of 7 organs (from `TestData/Labels.txt`, AMOS label scheme) × 10 images (5 CT, 5 MRI), computed Contrast-to-Noise Ratio and intensity std from the **ground-truth** segmentation masks, using a 5-voxel Euclidean-distance "surrounding" ring.

Published as an artifact ("Organ Contrast Metrics") with two data-quality flags surfaced rather than silently averaged over:
- `amos_0517` (MRI) uses raw, unnormalized scanner intensity units (values into the hundreds of thousands vs. ~50–800 for every other case) — a genuine property of that file, confirmed by inspecting the raw voxel range, not a computation bug.

## 3. TotalSegmentator accuracy vs. ground truth

`TestData/compute_totalseg_eval.py`: ran TotalSegmentator (free tasks, fast mode, GPU) on all 10 images, computed Dice and Hausdorff distance (exact + 95th percentile) against ground truth.

Published as a second artifact ("TotalSegmentator Accuracy"). Two findings investigated and confirmed, not just reported at face value:
- **Exact Hausdorff distance is outlier-dominated**: e.g. `amos_0004` right kidney has Dice 0.93 but HD = 187mm from a single spurious voxel; its HD95 is 2.3mm, consistent with the high Dice.
- **`amos_0005`'s "left kidney" ground truth is corrupted**: label 3 occupies 1,009 mL (a real kidney is ~120–150 mL) across 30 axial slices — verified via voxel-volume and bounding-box inspection, not assumed.

## 4. Combined results table

Merged the ground-truth-mask CNR, predicted-mask CNR, and Dice/Hausdorff CSVs into one file, grouped by organ (not by modality, per later correction) with per-organ average rows — `combine_results.py` → `Combined_Results.csv`.

## 5. Why window/level can't move CNR (and what can)

User asked why CNR "didn't really improve" after the contrast-optimization feature. Established mathematically: CNR = |mean_organ − mean_surround| / std_surround is **invariant under any linear rescale** (a·x+b) — which is exactly what window/level is — so no linear display transform can ever move it. This reframed the evaluation question: mean/std are *less* trustworthy for judging a linear enhancement (trivially inflatable by choice of scale), while CNR being flat is the mathematically correct result, not a flaw.

Follow-up: a transform that also **clips/saturates** (not just linear) breaks that invariance — which motivated the next feature.

## 6. Real, data-modifying enhancement

Added **background suppression** to the module: a new derived volume where every voxel *outside* the organ mask is pulled toward the volume's own minimum intensity by a tunable fraction (organ voxels untouched). Iterated on the UI per feedback:
- Initially a separate "Suppress Background" section/button → merged into "Optimize Contrast" (one click does both) per user request.
- Suppression range narrowed from 0–100% to 0–5%, default settled at 2.5% (via 1% → 2.5%), step 0.1%.
- Added: hide the segmentation overlay after optimizing; recenter all slice views on the organ's centroid (computed via IJK→RAS transform, verified against a synthetic volume with non-trivial spacing/origin).

This transform is *not* globally linear (organ untouched, background compressed), so CNR genuinely moves under it — verified: CNR improved for all 7 organs after adding 2.5% suppression, both on ground-truth and predicted masks.

## 7. Iteration 1 vs. Iteration 2 tracking

Built `Iterations/Iteration1/` (baseline, 0% suppression) and `Iterations/Iteration2/` (2.5% suppression) as parallel, comparable result sets.

**Bug caught and fixed:** Iteration 2's `gt_cnr` was accidentally computed from the *enhanced* image (using the ground-truth mask, but still the suppressed data) rather than the raw original — making it differ from Iteration 1's `gt_cnr` for the wrong reason. Fixed so `gt_cnr`/`gt_organ_mean`/`gt_organ_std` are always computed from the untouched original image + ground-truth mask only, making them iteration-independent by construction (`it1_gt_cnr == it2_gt_cnr` verified exactly, programmatically) — while `pred_cnr` (predicted mask) correctly keeps reflecting the enhancement.

Both scripts were later rebuilt to be **fully self-contained**: each runs TotalSegmentator itself (not dependent on a pre-existing segmentation folder), and each produces:
- One final `IterationN_Results.csv`
- `Segmentations/` (10 files) — full multi-label TotalSegmentator predictions
- `OrganMasks/` (70 files) — isolated per-organ binary predicted masks
- `EnhancedScenes/` (70 files) — the actual image data behind each row's `pred_cnr`
- `ENVIRONMENT.md` + `requirements.txt` — live-captured Python/package/GPU versions for reproducibility

**Honest finding, verified not assumed:** re-running TotalSegmentator's GPU inference is not perfectly bit-deterministic. `gt_*` columns reproduce exactly across runs (they never touch TotalSegmentator); `pred_*`/Hausdorff columns drift in the 3rd–4th significant figure between runs. Documented explicitly in each `ENVIRONMENT.md` rather than glossed over.

**Note on project volatility:** the folder structure (label filenames, segmentation storage location, CSV naming) was reorganized by the user multiple times *during* this work, independent of these requests. Each time, files assumed to exist had moved or been deleted — handled by re-surveying the actual filesystem state before proceeding rather than trusting prior assumptions, and reporting discrepancies rather than silently working around them.

## 8. Synthesis — the "hot takes"

- **CNR's rise after suppression is not strong evidence of a good enhancement** — pulling background toward the image floor mechanically increases CNR for any organ whose mean sits above background, which is nearly always true. The direction of the effect is close to guaranteed by the transform's construction, not evidence of careful tuning.
- **The segmentation failure mode is bimodal by organ size, not by modality**: kidneys (large, compact) score Dice 0.84–0.91 in both CT and MRI; adrenal glands (small) score ~0.44 and were missed entirely (zero predicted voxels) in 2 of 10 cases — traced to TotalSegmentator's fast-mode 3mm internal resampling erasing small structures. A single averaged Dice across all organs would hide this.
- **Ground truth isn't infallible**: the `amos_0005` left-kidney label is a verified annotation error sitting inside a public benchmark dataset, found by checking physical plausibility (volume in mL), not by trusting the label.
- **The weakest link in the overall system is segmentation reliability, not the enhancement algorithm.** The module does fail loudly on an empty/undetected segment rather than silently enhancing garbage — but there's no user-facing confidence signal distinguishing a well-segmented organ from a barely-detected one.

## Key files

| Path | What it is |
|---|---|
| `OrganAwareWindowing/OrganAwareWindowing.py` | The Slicer module |
| `TestData/compute_cnr.py`, `compute_totalseg_eval.py` | Original ground-truth-only metric scripts |
| `Iterations/Iteration1/run_iteration1.py` | Self-contained baseline (0% suppression) pipeline |
| `Iterations/Iteration2/run_iteration2.py` | Self-contained enhanced (2.5% suppression) pipeline |
| `Iterations/Iteration1/Iteration1_Results.csv`, `Iterations/Iteration2/Iteration2_Results.csv` | Final per-organ result tables |
| `Iterations/IterationN/ENVIRONMENT.md`, `requirements.txt` | Captured reproduction environment per iteration |

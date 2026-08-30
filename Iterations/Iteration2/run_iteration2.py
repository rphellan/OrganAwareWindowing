"""
Iteration 2: shows the effect of the OrganAwareWindowing module's default
2.5% background suppression on pred_cnr, against a 0%-suppression baseline
computed from the exact same TotalSegmentator prediction.

Self-contained: runs TotalSegmentator itself (the free "total"/"total_mr"
tasks) on every image in TestData/Images. Does not depend on Iteration 1's
folder or any pre-existing segmentation — nothing outside TestData/ needs
to exist beforehand.

gt_organ_mean / gt_organ_std / gt_cnr always use the original, unmodified
image and the ground-truth mask only (0% suppression) — a single shared set
of columns, since ground truth cannot be affected by the enhancement at all
and is identical regardless of iteration.

it1_* columns: predicted mask, 0% suppression (the "no enhancement"
baseline — computed from THIS run's own fresh segmentation, not Iteration
1's saved one, so expect small numeric drift vs. Iteration 1's file; see
the reproducibility note in ENVIRONMENT.md).
it2_* columns: predicted mask, 2.5% suppression (the module's actual
default enhancement).
it1_dice/it2_dice and it1_hausdorff*/it2_hausdorff* are duplicated across
both prefixes for column symmetry, but are always identical to each other
within a single run of this script — Dice/Hausdorff depend only on the
mask, never on the suppression fraction.

Running this script produces, in this same folder:
  Iteration2_Results.csv   the one final results table (columns above)
  Segmentations/            fresh multi-label TotalSegmentator predictions
  OrganMasks/               isolated per-organ binary predicted masks
  EnhancedScenes/           the actual 2.5%-suppressed image data used for
                            it2_pred_cnr — one per (image, organ). Unlike
                            Iteration 1's EnhancedScenes (byte-identical to
                            the raw images there, since that iteration is
                            0% by definition), these are genuinely
                            different from TestData/Images.
  ENVIRONMENT.md / requirements.txt   exact runtime environment, captured
                            live from this run.

No other CSV files are written.

Run with Slicer's bundled Python:
  Slicer-5.12.3-linux-amd64/bin/PythonSlicer run_iteration2.py
"""
import glob
import importlib.metadata
import os
import platform
import re
import sys

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.affines import apply_affine
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
from scipy.spatial import cKDTree
from totalsegmentator.map_to_binary import class_map
from totalsegmentator.python_api import totalsegmentator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
IMAGES_DIR = os.path.join(PROJECT_DIR, "TestData", "Images")
LABELS_DIR = os.path.join(PROJECT_DIR, "TestData", "Labels")
LABELS_TXT = os.path.join(PROJECT_DIR, "TestData", "Labels.txt")
PRED_SEG_DIR = os.path.join(SCRIPT_DIR, "Segmentations")
MASKS_DIR = os.path.join(SCRIPT_DIR, "OrganMasks")
SCENES_DIR = os.path.join(SCRIPT_DIR, "EnhancedScenes")
RESULTS_CSV = os.path.join(SCRIPT_DIR, "Iteration2_Results.csv")
ENVIRONMENT_MD = os.path.join(SCRIPT_DIR, "ENVIRONMENT.md")
REQUIREMENTS_TXT = os.path.join(SCRIPT_DIR, "requirements.txt")

SURROUND_RADIUS_VOXELS = 5
GT_SUPPRESSION_FRACTION = 0.0   # gt_cnr is always the raw-image baseline
IT1_SUPPRESSION_FRACTION = 0.0  # "no enhancement" comparison point
IT2_SUPPRESSION_FRACTION = 0.025  # OrganAwareWindowing module's current default (2.5%)

ORGAN_NAME_TO_TS_LABEL = {
    "right kidney": "kidney_right",
    "left kidney": "kidney_left",
    "gallbladder": "gallbladder",
    "esophagus": "esophagus",
    "right adrenal gland": "adrenal_gland_right",
    "left adrenal gland": "adrenal_gland_left",
    "duodenum": "duodenum",
}
MODALITY_TASK = {"CT": "total", "MRI": "total_mr"}
ORGAN_ORDER = list(ORGAN_NAME_TO_TS_LABEL.keys())

REQUIRED_PACKAGES = ["numpy", "scipy", "pandas", "nibabel", "totalsegmentator", "torch"]


def organ_slug(organName):
    return organName.replace(" ", "_")


def parse_labels_txt(path):
    organs = {}
    pattern = re.compile(r'"(\d+)"\s*:\s*"([^"]+)"')
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                organs[int(m.group(1))] = m.group(2)
    return organs


def infer_modality(imageFilename):
    if "_CT" in imageFilename.upper():
        return "CT"
    if "_MRI" in imageFilename.upper() or "_MR" in imageFilename.upper():
        return "MRI"
    return "Unknown"


def find_label_file(patientId):
    matches = sorted(glob.glob(os.path.join(LABELS_DIR, f"amos_{patientId}_*.nii.gz"))) + \
        sorted(glob.glob(os.path.join(LABELS_DIR, f"amos_{patientId}.nii.gz")))
    if not matches:
        raise FileNotFoundError(f"No label file for patient {patientId} in {LABELS_DIR}")
    return matches[0]


def find_seg_output_file(outputBase):
    for ext in (".nii.gz", ".nii"):
        candidate = outputBase + ext
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"TotalSegmentator did not produce an output at {outputBase}.nii[.gz]")


def run_totalsegmentator(imagePath, patientId, task):
    outputBase = os.path.join(PRED_SEG_DIR, f"amos_{patientId}_totalseg")
    totalsegmentator(
        input=imagePath,
        output=outputBase,
        task=task,
        roi_subset=list(ORGAN_NAME_TO_TS_LABEL.values()),
        fast=True,
        ml=True,
        device="gpu",
        quiet=True,
    )
    return find_seg_output_file(outputBase)


def apply_background_suppression(imageArray, organMask, fraction):
    if fraction == 0.0:
        return imageArray.astype(np.float64, copy=False)
    floorValue = float(imageArray.min())
    enhanced = imageArray.astype(np.float64, copy=True)
    background = ~organMask
    enhanced[background] = enhanced[background] * (1.0 - fraction) + floorValue * fraction
    return enhanced


def compute_cnr_and_std(enhancedArray, organMask):
    nOrgan = int(organMask.sum())
    if nOrgan == 0:
        return dict(organ_mean=np.nan, organ_std=np.nan, cnr=np.nan)

    distanceFromOrgan = distance_transform_edt(~organMask)
    surroundingMask = (~organMask) & (distanceFromOrgan <= SURROUND_RADIUS_VOXELS)
    nSurround = int(surroundingMask.sum())

    organValues = enhancedArray[organMask]
    organMean = float(organValues.mean())
    organStd = float(organValues.std())

    if nSurround == 0:
        return dict(organ_mean=organMean, organ_std=organStd, cnr=np.nan)

    surroundValues = enhancedArray[surroundingMask]
    surroundMean = float(surroundValues.mean())
    surroundStd = float(surroundValues.std())
    cnr = np.nan if surroundStd == 0 else abs(organMean - surroundMean) / surroundStd

    return dict(organ_mean=organMean, organ_std=organStd, cnr=cnr)


def dice_coefficient(a, b):
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return np.nan
    return 2.0 * np.logical_and(a, b).sum() / denom


def surface_points_mm(mask, affine):
    if mask.sum() == 0:
        return None
    structure = generate_binary_structure(mask.ndim, 1)
    eroded = binary_erosion(mask, structure=structure, border_value=0)
    border = mask & ~eroded
    coords = np.argwhere(border)
    if coords.shape[0] == 0:
        coords = np.argwhere(mask)
    return apply_affine(affine, coords)


def hausdorff_distances(maskPred, maskGt, affine):
    ptsPred = surface_points_mm(maskPred, affine)
    ptsGt = surface_points_mm(maskGt, affine)
    if ptsPred is None or ptsGt is None:
        return np.nan, np.nan
    treeGt = cKDTree(ptsGt)
    treePred = cKDTree(ptsPred)
    dPredToGt, _ = treeGt.query(ptsPred)
    dGtToPred, _ = treePred.query(ptsGt)
    hd = float(max(dPredToGt.max(), dGtToPred.max()))
    hd95 = float(max(np.percentile(dPredToGt, 95), np.percentile(dGtToPred, 95)))
    return hd, hd95


def build_results_table(rows):
    df = pd.DataFrame(rows)
    metricCols = ["gt_organ_mean", "gt_organ_std", "gt_cnr",
                  "it1_pred_organ_mean", "it1_pred_organ_std", "it1_pred_cnr",
                  "it1_dice", "it1_hausdorff_mm", "it1_hausdorff95_mm",
                  "it2_pred_organ_mean", "it2_pred_organ_std", "it2_pred_cnr",
                  "it2_dice", "it2_hausdorff_mm", "it2_hausdorff95_mm"]
    out = []
    for organ in ORGAN_ORDER:
        group = df[df["organ"] == organ]
        row = {"organ": organ}
        for col in metricCols:
            row[col] = round(float(group[col].mean(skipna=True)), 3)
        out.append(row)
    return pd.DataFrame(out, columns=["organ"] + metricCols)


def write_environment_description():
    lines = []
    lines.append("# Reproduction environment — Iteration 2\n")
    lines.append("Captured automatically at the start of `run_iteration2.py`. Use the same ")
    lines.append("Python interpreter and package versions listed here to reproduce `Iteration2_Results.csv`.\n")

    lines.append("## Interpreter\n")
    lines.append(f"- Python: `{platform.python_version()}`")
    lines.append(f"- Executable: `{sys.executable}`")
    lines.append(f"- Platform: `{platform.platform()}`\n")

    lines.append("## Required packages\n")
    lines.append("| Package | Version |")
    lines.append("|---|---|")
    versions = {}
    for pkg in REQUIRED_PACKAGES:
        try:
            v = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            v = "NOT INSTALLED"
        versions[pkg] = v
        lines.append(f"| {pkg} | {v} |")
    lines.append("")

    try:
        import torch
        cudaAvailable = torch.cuda.is_available()
        lines.append("## GPU\n")
        lines.append(f"- CUDA available: `{cudaAvailable}`")
        if cudaAvailable:
            lines.append(f"- Device: `{torch.cuda.get_device_name(0)}`")
        lines.append("")
    except ImportError:
        pass

    lines.append("## How to run\n")
    lines.append("This script must run under Slicer's bundled Python (not system Python), ")
    lines.append("which already ships nibabel/numpy/scipy/pandas. It is self-contained: it runs ")
    lines.append("TotalSegmentator itself (downloading the free `total`/`total_mr` model weights ")
    lines.append("on first use if not already cached) and writes its own `Segmentations/` folder ")
    lines.append("— nothing needs to exist beforehand except TestData/.\n")
    lines.append("```")
    lines.append("Slicer-5.12.3-linux-amd64/bin/PythonSlicer run_iteration2.py")
    lines.append("```\n")
    lines.append("Expects, relative to the project root: `TestData/Images/`, `TestData/Labels/`, ")
    lines.append("`TestData/Labels.txt`. Nothing else.\n")
    lines.append("## Reproducibility note\n")
    lines.append("`gt_organ_mean`, `gt_organ_std`, and `gt_cnr` are exactly reproducible on any ")
    lines.append("re-run — they only depend on the ground-truth masks and raw image data, both ")
    lines.append("static inputs. `it1_*`/`it2_*` columns depend on a fresh TotalSegmentator GPU ")
    lines.append("inference pass each run, which is not perfectly bit-deterministic: expect ")
    lines.append("agreement to 3-4 significant figures, not exact equality, across separate runs ")
    lines.append("(even on the same machine). `it1_*` and `it2_*` within the SAME run are always ")
    lines.append("computed from the identical segmentation, so it1_dice == it2_dice etc. exactly.\n")

    with open(ENVIRONMENT_MD, "w") as f:
        f.write("\n".join(lines))

    with open(REQUIREMENTS_TXT, "w") as f:
        for pkg, v in versions.items():
            if v != "NOT INSTALLED":
                f.write(f"{pkg}=={v}\n")

    print(f"Wrote {ENVIRONMENT_MD}")
    print(f"Wrote {REQUIREMENTS_TXT}")


def main():
    write_environment_description()

    os.makedirs(PRED_SEG_DIR, exist_ok=True)
    os.makedirs(MASKS_DIR, exist_ok=True)
    os.makedirs(SCENES_DIR, exist_ok=True)

    organsOfInterest = parse_labels_txt(LABELS_TXT)
    imagePaths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.nii.gz")))

    rows = []

    for imagePath in imagePaths:
        imageFilename = os.path.basename(imagePath)
        patientId = imageFilename.split("_")[1]
        modality = infer_modality(imageFilename)
        task = MODALITY_TASK[modality]
        print(f"Processing {imageFilename} ({modality}) ...")

        imageNii = nib.load(imagePath)
        imageArray = imageNii.get_fdata()
        sourceDtype = imageNii.get_data_dtype()

        labelNii = nib.load(find_label_file(patientId))
        gtData = labelNii.get_fdata().astype(np.int32)
        assert labelNii.shape == imageNii.shape, f"GT/image shape mismatch for {imageFilename}"

        print(f"  Segmenting with TotalSegmentator (task={task}) ...")
        predSegPath = run_totalsegmentator(imagePath, patientId, task)
        predNii = nib.load(predSegPath)
        predData = predNii.get_fdata().astype(np.int32)
        assert predNii.shape == imageNii.shape, f"pred/image shape mismatch for {imageFilename}"

        tsLabelIds = {v: k for k, v in class_map[task].items()}

        for gtLabelId, organName in sorted(organsOfInterest.items()):
            slug = organ_slug(organName)

            gtMask = gtData == gtLabelId
            statsGt = compute_cnr_and_std(
                apply_background_suppression(imageArray, gtMask, GT_SUPPRESSION_FRACTION), gtMask)

            tsLabelName = ORGAN_NAME_TO_TS_LABEL[organName]
            predLabelId = tsLabelIds[tsLabelName]
            predMask = predData == predLabelId

            statsIt1 = compute_cnr_and_std(
                apply_background_suppression(imageArray, predMask, IT1_SUPPRESSION_FRACTION), predMask)
            enhancedIt2 = apply_background_suppression(imageArray, predMask, IT2_SUPPRESSION_FRACTION)
            statsIt2 = compute_cnr_and_std(enhancedIt2, predMask)

            # predicted-mask organ mask, isolated
            maskPath = os.path.join(MASKS_DIR, f"amos_{patientId}_{slug}_pred_mask.nii.gz")
            nib.save(nib.Nifti1Image(predMask.astype(np.uint8), imageNii.affine), maskPath)

            # the actual 2.5%-suppressed image data this row's it2_pred_cnr was computed from
            scenePath = os.path.join(SCENES_DIR, f"amos_{patientId}_{slug}_pred_scene.nii.gz")
            nib.save(nib.Nifti1Image(enhancedIt2.astype(sourceDtype), imageNii.affine), scenePath)

            d = dice_coefficient(predMask, gtMask)
            hd, hd95 = hausdorff_distances(predMask, gtMask, labelNii.affine)

            rows.append(dict(
                image=imageFilename, patient_id=patientId, modality=modality, organ=organName,
                gt_organ_mean=statsGt["organ_mean"], gt_organ_std=statsGt["organ_std"], gt_cnr=statsGt["cnr"],
                it1_pred_organ_mean=statsIt1["organ_mean"], it1_pred_organ_std=statsIt1["organ_std"],
                it1_pred_cnr=statsIt1["cnr"], it1_dice=d, it1_hausdorff_mm=hd, it1_hausdorff95_mm=hd95,
                it2_pred_organ_mean=statsIt2["organ_mean"], it2_pred_organ_std=statsIt2["organ_std"],
                it2_pred_cnr=statsIt2["cnr"], it2_dice=d, it2_hausdorff_mm=hd, it2_hausdorff95_mm=hd95,
            ))

    results = build_results_table(rows)
    results.to_csv(RESULTS_CSV, index=False)

    segCount = len(glob.glob(os.path.join(PRED_SEG_DIR, "*.nii*")))
    maskCount = len(glob.glob(os.path.join(MASKS_DIR, "*.nii.gz")))
    sceneCount = len(glob.glob(os.path.join(SCENES_DIR, "*.nii.gz")))
    print(f"\nSaved {RESULTS_CSV} ({len(results)} rows)")
    print(f"Saved {segCount} segmentations to {PRED_SEG_DIR}")
    print(f"Saved {maskCount} organ masks to {MASKS_DIR}")
    print(f"Saved {sceneCount} scenes to {SCENES_DIR}")

    pd.set_option("display.width", 250)
    print("\n--- Iteration 2 per-organ averages (n=10 each) ---")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

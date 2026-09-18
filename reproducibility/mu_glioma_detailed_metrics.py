"""Uniform detailed evaluation helpers for all MU-Glioma segmentation runs."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from surface_distance import metrics as surface_distance_metrics


SURFACE_DICE_TOLERANCE_MM = 1.0
LESION_CONNECTIVITY = 3  # scipy rank-3/connectivity-3 = 26-connected components

TARGET_DEFINITIONS = {
    "Non-enhancing tumor core": {1},
    "FLAIR hyperintensity / edema": {2},
    "Enhancing tissue": {3},
    "Resection cavity": {4},
    "Tumor-related region (1-3)": {1, 2, 3},
    "All postoperative regions (1-4)": {1, 2, 3, 4},
}

REQUIRED_PER_CASE_COLUMNS = [
    "run_id",
    "model",
    "cv_fold",
    "split_seed",
    "training_seed",
    "case_id",
    "patient_id",
    "target",
    "tp",
    "fp",
    "fn",
    "dice",
    "iou",
    "precision",
    "recall",
    "gt_present",
    "pred_present",
    "truth_voxels",
    "predicted_voxels",
    "voxel_spacing_x_mm",
    "voxel_spacing_y_mm",
    "voxel_spacing_z_mm",
    "surface_distance_defined",
    "hd95_mm",
    "average_surface_distance_mm",
    "average_surface_distance_gt_to_pred_mm",
    "average_surface_distance_pred_to_gt_mm",
    "surface_dice_1mm",
    "reference_lesions",
    "predicted_lesions",
    "detected_reference_lesions",
    "missed_reference_lesions",
    "false_positive_lesions",
    "lesion_sensitivity",
]

METRIC_DEFINITIONS = {
    "version": "mu_glioma_uniform_metrics_v2_fixed_reference_subset",
    "targets": {key: sorted(value) for key, value in TARGET_DEFINITIONS.items()},
    "overlap": {
        "dice": "2TP/(2TP+FP+FN)",
        "iou": "TP/(TP+FP+FN)",
        "precision": "TP/(TP+FP)",
        "recall": "TP/(TP+FN)",
        "both_reference_and_prediction_empty": "Dice/IoU/precision/recall are NaN and excluded from means",
        "reference_empty_prediction_present": "Dice/IoU/precision are 0; recall is NaN",
        "aggregation_subset": "all overlap summaries use the fixed reference-present scans only; reference-absent scans are reported separately",
    },
    "surface": {
        "implementation": "google-deepmind surface-distance 0.1 with physical NIfTI voxel spacing",
        "hd95_mm": "area-weighted symmetric robust Hausdorff distance at the 95th percentile",
        "average_surface_distance_mm": "unweighted mean of the two area-weighted directed average surface distances",
        "surface_dice_tolerance_mm": SURFACE_DICE_TOLERANCE_MM,
        "both_nonempty_required_for_distances": True,
        "one_empty_policy": "HD95/average distances NaN; surface Dice 0",
        "both_empty_policy": "all surface metrics NaN",
    },
    "lesion": {
        "connectivity": "26-connected components in 3D",
        "detection_rule": "a reference component is detected when at least one predicted voxel overlaps it",
        "false_positive_rule": "a predicted component is false-positive when it overlaps no reference voxel",
        "absent_reference_sensitivity": "NaN",
    },
    "absent_reference": {
        "false_positive_scan": "reference target absent and at least one target voxel predicted",
        "confidence_interval": "95% Wilson binomial interval across distinct scans",
        "independence_note": "seed repeats are not counted as new independent patients",
    },
}


def validate_discrete_label_volume(
    array: np.ndarray,
    context: str = "label volume",
    allowed_labels=(0, 1, 2, 3, 4),
) -> np.ndarray:
    """Validate before casting so overflow cannot hide invalid predictions."""
    array = np.asanyarray(array)
    assert np.issubdtype(array.dtype, np.number), (
        f"{context}: non-numeric dtype {array.dtype}"
    )
    assert np.all(np.isfinite(array)), f"{context}: NaN or infinite values"
    assert np.all(array == np.rint(array)), f"{context}: non-integer label values"
    observed = {int(value) for value in np.unique(array)}
    allowed = set(int(value) for value in allowed_labels)
    assert observed.issubset(allowed), (
        f"{context}: unexpected labels {sorted(observed - allowed)}"
    )
    return array.astype(np.uint8, copy=False)


def binary_overlap_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    truth = np.asarray(truth, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    tp = int(np.logical_and(prediction, truth).sum())
    fp = int(np.logical_and(prediction, ~truth).sum())
    fn = int(np.logical_and(~prediction, truth).sum())
    gt_present = bool(truth.any())
    pred_present = bool(prediction.any())
    if not gt_present and not pred_present:
        dice = iou = precision = recall = np.nan
    else:
        dice = (2 * tp) / max(2 * tp + fp + fn, 1)
        iou = tp / max(tp + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1) if gt_present else np.nan
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "gt_present": gt_present,
        "pred_present": pred_present,
        "truth_voxels": int(truth.sum()),
        "predicted_voxels": int(prediction.sum()),
    }


def surface_and_lesion_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    spacing_mm,
) -> dict:
    truth = np.asarray(truth, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    spacing_mm = tuple(float(value) for value in spacing_mm)
    gt_present = bool(truth.any())
    pred_present = bool(prediction.any())

    if gt_present and pred_present:
        distances = surface_distance_metrics.compute_surface_distances(
            truth, prediction, spacing_mm=spacing_mm
        )
        directed_asd = surface_distance_metrics.compute_average_surface_distance(
            distances
        )
        hd95_mm = float(
            surface_distance_metrics.compute_robust_hausdorff(distances, 95.0)
        )
        asd_gt_to_pred = float(directed_asd[0])
        asd_pred_to_gt = float(directed_asd[1])
        average_surface_distance = (asd_gt_to_pred + asd_pred_to_gt) / 2.0
        surface_dice = float(
            surface_distance_metrics.compute_surface_dice_at_tolerance(
                distances, tolerance_mm=SURFACE_DICE_TOLERANCE_MM
            )
        )
        surface_distance_defined = True
    elif gt_present or pred_present:
        hd95_mm = np.nan
        asd_gt_to_pred = np.nan
        asd_pred_to_gt = np.nan
        average_surface_distance = np.nan
        surface_dice = 0.0
        surface_distance_defined = False
    else:
        hd95_mm = np.nan
        asd_gt_to_pred = np.nan
        asd_pred_to_gt = np.nan
        average_surface_distance = np.nan
        surface_dice = np.nan
        surface_distance_defined = False

    structure = ndimage.generate_binary_structure(3, LESION_CONNECTIVITY)
    truth_components, reference_lesions = ndimage.label(truth, structure=structure)
    prediction_components, predicted_lesions = ndimage.label(
        prediction, structure=structure
    )
    detected_reference_lesions = sum(
        bool(prediction[truth_components == component].any())
        for component in range(1, reference_lesions + 1)
    )
    false_positive_lesions = sum(
        not bool(truth[prediction_components == component].any())
        for component in range(1, predicted_lesions + 1)
    )
    missed_reference_lesions = reference_lesions - detected_reference_lesions
    lesion_sensitivity = (
        detected_reference_lesions / reference_lesions
        if reference_lesions
        else np.nan
    )
    return {
        "voxel_spacing_x_mm": spacing_mm[0],
        "voxel_spacing_y_mm": spacing_mm[1],
        "voxel_spacing_z_mm": spacing_mm[2],
        "surface_distance_defined": surface_distance_defined,
        "hd95_mm": hd95_mm,
        "average_surface_distance_mm": average_surface_distance,
        "average_surface_distance_gt_to_pred_mm": asd_gt_to_pred,
        "average_surface_distance_pred_to_gt_mm": asd_pred_to_gt,
        "surface_dice_1mm": surface_dice,
        "reference_lesions": int(reference_lesions),
        "predicted_lesions": int(predicted_lesions),
        "detected_reference_lesions": int(detected_reference_lesions),
        "missed_reference_lesions": int(missed_reference_lesions),
        "false_positive_lesions": int(false_positive_lesions),
        "lesion_sensitivity": lesion_sensitivity,
    }


def evaluate_case_targets(
    truth_labels: np.ndarray,
    prediction_labels: np.ndarray,
    spacing_mm,
    identity: dict,
) -> list[dict]:
    rows = []
    for target_name, target_labels in TARGET_DEFINITIONS.items():
        truth = np.isin(truth_labels, list(target_labels))
        prediction = np.isin(prediction_labels, list(target_labels))
        row = dict(identity)
        row["target"] = target_name
        row.update(binary_overlap_metrics(truth, prediction))
        row.update(surface_and_lesion_metrics(truth, prediction, spacing_mm))
        rows.append(row)
    return rows


def metrics_from_counts(tp: int, fp: int, fn: int):
    if tp == 0 and fp == 0 and fn == 0:
        return np.nan, np.nan, np.nan, np.nan
    dice = (2 * tp) / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1) if (tp + fn) > 0 else np.nan
    return dice, iou, precision, recall


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054):
    if total == 0:
        return np.nan, np.nan
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        rate * (1 - rate) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def build_uniform_summaries(scores: pd.DataFrame):
    missing = [column for column in REQUIRED_PER_CASE_COLUMNS if column not in scores]
    assert not missing, f"Missing required per-case columns: {missing}"

    patient_rows = []
    for (patient_id, target), group in scores.groupby(
        ["patient_id", "target"], sort=True
    ):
        present = group[group.gt_present]
        absent = group[~group.gt_present]
        tp, fp, fn = (
            int(present.tp.sum()),
            int(present.fp.sum()),
            int(present.fn.sum()),
        )
        dice, iou, precision, recall = metrics_from_counts(tp, fp, fn)
        reference_lesions = int(present.reference_lesions.sum())
        detected_lesions = int(present.detected_reference_lesions.sum())
        patient_rows.append(
            {
                "run_id": group.run_id.iloc[0],
                "model": group.model.iloc[0],
                "cv_fold": int(group.cv_fold.iloc[0]),
                "split_seed": int(group.split_seed.iloc[0]),
                "training_seed": int(group.training_seed.iloc[0]),
                "patient_id": patient_id,
                "target": target,
                "scan_count": int(group.case_id.nunique()),
                "reference_present_scan_count": int(present.case_id.nunique()),
                "reference_absent_scan_count": int(absent.case_id.nunique()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "pooled_dice": dice,
                "pooled_iou": iou,
                "pooled_precision": precision,
                "pooled_recall": recall,
                "mean_scan_dice": float(present.dice.mean()),
                "median_scan_hd95_mm": float(present.hd95_mm.median()),
                "mean_scan_average_surface_distance_mm": float(
                    present.average_surface_distance_mm.mean()
                ),
                "mean_scan_surface_dice_1mm": float(
                    present.surface_dice_1mm.mean()
                ),
                "reference_lesions": reference_lesions,
                "detected_reference_lesions": detected_lesions,
                "missed_reference_lesions": int(
                    present.missed_reference_lesions.sum()
                ),
                "false_positive_lesions_on_reference_present_scans": int(
                    present.false_positive_lesions.sum()
                ),
                "false_positive_scans_when_reference_absent": int(
                    absent.pred_present.sum()
                ),
                "false_positive_lesions_when_reference_absent": int(
                    absent.false_positive_lesions.sum()
                ),
                "lesion_sensitivity": (
                    detected_lesions / reference_lesions
                    if reference_lesions
                    else np.nan
                ),
            }
        )
    patient_scores = pd.DataFrame(patient_rows)

    summary_rows = []
    absent_rows = []
    for target_name in TARGET_DEFINITIONS:
        subset = scores[scores.target == target_name]
        present = subset[subset.gt_present]
        dice_values = present.dice.dropna()
        q1, q3 = dice_values.quantile([0.25, 0.75])
        reference_lesions = int(present.reference_lesions.sum())
        detected_lesions = int(present.detected_reference_lesions.sum())
        absent = subset[~subset.gt_present]
        false_positive_scans = int(absent.pred_present.sum())
        ci_low, ci_high = wilson_interval(false_positive_scans, len(absent))
        summary_rows.append(
            {
                "run_id": subset.run_id.iloc[0],
                "model": subset.model.iloc[0],
                "cv_fold": int(subset.cv_fold.iloc[0]),
                "split_seed": int(subset.split_seed.iloc[0]),
                "training_seed": int(subset.training_seed.iloc[0]),
                "target": target_name,
                "mean_dice": float(dice_values.mean()),
                "sd_dice": float(dice_values.std(ddof=1)),
                "median_dice": float(dice_values.median()),
                "q1_dice": float(q1),
                "q3_dice": float(q3),
                "mean_iou": float(present.iou.mean()),
                "mean_precision": float(present.precision.mean()),
                "mean_recall": float(present.recall.mean()),
                "median_hd95_mm": float(present.hd95_mm.median()),
                "mean_average_surface_distance_mm": float(
                    present.average_surface_distance_mm.mean()
                ),
                "mean_surface_dice_1mm": float(
                    present.surface_dice_1mm.mean()
                ),
                "reference_lesions": reference_lesions,
                "detected_reference_lesions": detected_lesions,
                "missed_reference_lesions": int(
                    present.missed_reference_lesions.sum()
                ),
                "false_positive_lesions_on_reference_present_scans": int(
                    present.false_positive_lesions.sum()
                ),
                "lesion_sensitivity": (
                    detected_lesions / reference_lesions
                    if reference_lesions
                    else np.nan
                ),
                "validation_scans": int(subset.case_id.nunique()),
                "validation_patients": int(subset.patient_id.nunique()),
                "reference_present_scans": int(present.case_id.nunique()),
                "reference_present_patients": int(present.patient_id.nunique()),
                "absent_reference_scans": int(len(absent)),
                "absent_reference_patients": int(absent.patient_id.nunique()),
                "absent_reference_false_positive_scans": false_positive_scans,
                "absent_reference_false_positive_rate": (
                    false_positive_scans / len(absent) if len(absent) else np.nan
                ),
                "absent_reference_false_positive_rate_ci95_low": ci_low,
                "absent_reference_false_positive_rate_ci95_high": ci_high,
            }
        )
        absent_rows.append(
            {
                "run_id": subset.run_id.iloc[0],
                "model": subset.model.iloc[0],
                "cv_fold": int(subset.cv_fold.iloc[0]),
                "training_seed": int(subset.training_seed.iloc[0]),
                "target": target_name,
                "absent_reference_scans": int(len(absent)),
                "absent_reference_patients": int(absent.patient_id.nunique()),
                "false_positive_scans": false_positive_scans,
                "false_positive_scan_rate": (
                    false_positive_scans / len(absent) if len(absent) else np.nan
                ),
                "false_positive_scan_rate_ci95_low": ci_low,
                "false_positive_scan_rate_ci95_high": ci_high,
                "mean_predicted_voxels_when_absent": float(
                    absent.predicted_voxels.mean()
                ),
                "median_predicted_voxels_when_absent": float(
                    absent.predicted_voxels.median()
                ),
                "false_positive_lesions_when_absent": int(
                    absent.false_positive_lesions.sum()
                ),
            }
        )
    return patient_scores, pd.DataFrame(summary_rows), pd.DataFrame(absent_rows)


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_prediction_manifest(prediction_dir, validation_cases: list[dict]):
    case_lookup = {case["id"]: case for case in validation_cases}
    rows = []
    for path in sorted(Path(prediction_dir).glob("*.nii.gz")):
        case_id = path.name.removesuffix(".nii.gz")
        case = case_lookup.get(case_id)
        rows.append(
            {
                "case_id": case_id,
                "patient_id": case["patient"] if case else "",
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)

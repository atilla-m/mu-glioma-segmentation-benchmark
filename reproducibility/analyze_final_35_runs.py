#!/usr/bin/env python3
"""Prespecified final analysis for the validated MU-Glioma-Post experiment.

The primary inferential unit is the patient. Primary model estimates concatenate
the five seed-2026 outer folds, giving one out-of-fold score per patient and
architecture. Fold-1 seed repeats are analyzed separately and never inflate n.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np
import pandas as pd
import scipy
from scipy import stats


PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parent
ZIP_ROOT = PACKAGE / "results" / "validated_zips"
TRACKER_PATH = PACKAGE / "results" / "progress" / "MU_Glioma_35_Run_Progress.csv"
OUTPUT = PACKAGE / "results" / "final_analysis"
FIGURES = PACKAGE / "manuscript" / "figures"
DATASET_ROOT = ROOT / "MU-Glioma-Post"
LEGACY_SPLIT = (
    ROOT
    / "MU_Glioma_Keras3DUNet-20260821T114449Z-1-001"
    / "MU_Glioma_Keras3DUNet"
    / "patient_split.json"
)
DEVELOPMENT_HOLDOUT = PACKAGE / "protocol" / "development_holdout_patients.json"

BOOTSTRAP_RESAMPLES = 100_000
BOOTSTRAP_SEED = 2026
PRIMARY_TARGET = "Tumor-related region (1-3)"

MODEL_ORDER = [
    "Standard 3D U-Net",
    "Residual 3D U-Net",
    "Controlled 3D V-Net",
    "Controlled 3D U-Net++",
    "nnU-Net v2 3D fullres",
]
MODEL_LABELS = {
    "Standard 3D U-Net": "3D U-Net",
    "Residual 3D U-Net": "Residual U-Net",
    "Controlled 3D V-Net": "V-Net",
    "Controlled 3D U-Net++": "U-Net++",
    "nnU-Net v2 3D fullres": "nnU-Net",
}
MODEL_COLORS = {
    "Standard 3D U-Net": "#577590",
    "Residual 3D U-Net": "#277da1",
    "Controlled 3D V-Net": "#43aa8b",
    "Controlled 3D U-Net++": "#f8961e",
    "nnU-Net v2 3D fullres": "#9b5de5",
}
TARGET_ORDER = [
    "Non-enhancing tumor core",
    "FLAIR hyperintensity / edema",
    "Enhancing tissue",
    "Resection cavity",
    "Tumor-related region (1-3)",
    "All postoperative regions (1-4)",
]
TARGET_SHORT = {
    "Non-enhancing tumor core": "Non-enhancing core",
    "FLAIR hyperintensity / edema": "FLAIR/edema",
    "Enhancing tissue": "Enhancing tissue",
    "Resection cavity": "Resection cavity",
    "Tumor-related region (1-3)": "Tumor union (1–3)",
    "All postoperative regions (1-4)": "All regions (1–4)",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zip_path(run_id: str) -> Path:
    matches = sorted((ZIP_ROOT / run_id).glob("*.zip"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one ZIP for {run_id}, found {len(matches)}")
    return matches[0]


def read_zip_member(zpath: Path, suffix: str) -> bytes:
    with zipfile.ZipFile(zpath) as archive:
        matches = [name for name in archive.namelist() if name.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one member ending {suffix!r} in {zpath}, found {matches}"
            )
        return archive.read(matches[0])


def read_zip_csv(zpath: Path, suffix: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(read_zip_member(zpath, suffix)))


def read_zip_json(zpath: Path, suffix: str) -> dict:
    return json.loads(read_zip_member(zpath, suffix).decode("utf-8"))


def bootstrap_ci(values, statistic="mean", seed=BOOTSTRAP_SEED):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    chunk = 5_000
    fn = np.mean if statistic == "mean" else np.median
    for start in range(0, BOOTSTRAP_RESAMPLES, chunk):
        stop = min(start + chunk, BOOTSTRAP_RESAMPLES)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        samples[start:stop] = fn(values[indices], axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]).tolist())


def wilson_ci(successes: int, total: int, alpha: float = 0.05):
    if total == 0:
        return math.nan, math.nan
    z = stats.norm.ppf(1 - alpha / 2)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def rank_biserial_paired(a, b):
    """Matched-pairs rank-biserial correlation for differences a - b."""
    differences = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    differences = differences[np.isfinite(differences)]
    nonzero = differences[differences != 0]
    if not nonzero.size:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    return float((positive - negative) / (positive + negative))


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        candidate = (m - rank) * p_values[index]
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return adjusted


def paired_comparison(frame, comparator, reference, seed):
    pivot = frame.pivot(index="patient_id", columns="architecture", values="primary_dice")
    paired = pivot[[comparator, reference]].dropna()
    differences = paired[comparator].to_numpy() - paired[reference].to_numpy()
    ci_low, ci_high = bootstrap_ci(differences, statistic="median", seed=seed)
    test = stats.wilcoxon(
        paired[comparator],
        paired[reference],
        zero_method="pratt",
        correction=False,
        alternative="two-sided",
        method="auto",
    )
    return {
        "comparison": f"{MODEL_LABELS[comparator]} vs {MODEL_LABELS[reference]}",
        "comparator": comparator,
        "reference": reference,
        "n_patients": len(paired),
        "median_paired_difference": float(np.median(differences)),
        "bootstrap_ci95_low": ci_low,
        "bootstrap_ci95_high": ci_high,
        "mean_paired_difference": float(np.mean(differences)),
        "rank_biserial": rank_biserial_paired(paired[comparator], paired[reference]),
        "wilcoxon_statistic": float(test.statistic),
        "p_value": float(test.pvalue),
    }


def normalize_bool(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False}).astype(bool)


def collect_data():
    tracker = pd.read_csv(TRACKER_PATH)
    if len(tracker) != 35 or not (tracker["archive_validation"] == "PASS").all():
        raise RuntimeError("The final analysis requires 35/35 validated archives")

    patients = []
    cases = []
    run_records = []
    for row in tracker.itertuples(index=False):
        zpath = zip_path(row.run_id)
        patient_frame = read_zip_csv(zpath, "outer_test_per_patient_metrics.csv")
        case_frame = read_zip_csv(zpath, "outer_test_per_case_metrics.csv")
        summary = read_zip_json(zpath, "final_summary.json")
        efficiency = read_zip_json(zpath, "efficiency.json")

        patient_frame["architecture"] = row.architecture
        patient_frame["analysis_role"] = row.analysis_role
        patient_frame["platform"] = row.platform
        case_frame["architecture"] = row.architecture
        case_frame["analysis_role"] = row.analysis_role
        case_frame["platform"] = row.platform
        case_frame["gt_present"] = normalize_bool(case_frame["gt_present"])
        case_frame["pred_present"] = normalize_bool(case_frame["pred_present"])
        case_frame["surface_distance_defined"] = normalize_bool(
            case_frame["surface_distance_defined"]
        )
        patients.append(patient_frame)
        cases.append(case_frame)

        training_seconds = efficiency.get(
            "training_wall_seconds", efficiency.get("training_and_tuning_wall_seconds")
        )
        peak_bytes = efficiency.get("peak_gpu_memory_bytes")
        peak_mib = efficiency.get("peak_observed_gpu_memory_mib")
        if peak_bytes is not None:
            peak_gib = float(peak_bytes) / 1024**3
        elif peak_mib is not None:
            peak_gib = float(peak_mib) / 1024
        else:
            peak_gib = math.nan
        inference_per_case = efficiency.get("outer_test_inference_mean_seconds_per_case")
        if inference_per_case is None:
            inference_total = efficiency.get("outer_test_inference_wall_seconds")
            inference_per_case = (
                float(inference_total) / int(row.outer_test_cases)
                if inference_total is not None
                else math.nan
            )
        run_records.append(
            {
                "run_id": row.run_id,
                "architecture": row.architecture,
                "analysis_role": row.analysis_role,
                "fold": int(row.cv_fold),
                "training_seed": int(row.training_seed),
                "platform": row.platform,
                "completed_epochs": int(summary["completed_epochs"]),
                "best_epoch": int(summary.get("best_epoch", summary.get("best_saved_epoch"))),
                "stop_reason": summary["stop_reason"],
                "training_hours": float(training_seconds) / 3600,
                "total_parameters": int(efficiency["total_parameters"]),
                "peak_gpu_memory_gib": peak_gib,
                "inference_seconds_per_case": float(inference_per_case),
                "archive_mib": zpath.stat().st_size / 1024**2,
                "zip_path": str(zpath.relative_to(ROOT)),
                "zip_sha256": sha256(zpath),
            }
        )

    patient_data = pd.concat(patients, ignore_index=True)
    case_data = pd.concat(cases, ignore_index=True)
    runs = pd.DataFrame(run_records)
    return tracker, patient_data, case_data, runs


def validate_primary(patient_data, case_data):
    primary_patients = patient_data.query(
        "analysis_role == 'primary_five_fold_cross_validation'"
    )
    primary_cases = case_data.query("analysis_role == 'primary_five_fold_cross_validation'")
    errors = []
    for model in MODEL_ORDER:
        p = primary_patients[primary_patients.architecture == model]
        c = primary_cases[primary_cases.architecture == model]
        if p.patient_id.nunique() != 203:
            errors.append(f"{model}: {p.patient_id.nunique()} unique patients, expected 203")
        if c.case_id.nunique() != 594:
            errors.append(f"{model}: {c.case_id.nunique()} unique cases, expected 594")
        if p.run_id.nunique() != 5:
            errors.append(f"{model}: {p.run_id.nunique()} primary runs, expected 5")
        duplicates = p.groupby(["patient_id", "target"]).size().max()
        if duplicates != 1:
            errors.append(f"{model}: patient-target maximum multiplicity {duplicates}")
    expected_cases = None
    for model in MODEL_ORDER:
        observed = set(primary_cases.loc[primary_cases.architecture == model, "case_id"])
        if expected_cases is None:
            expected_cases = observed
        elif observed != expected_cases:
            errors.append(f"{model}: outer-test case identities differ")
    if errors:
        raise RuntimeError("Primary validation failed:\n" + "\n".join(errors))
    return primary_patients.copy(), primary_cases.copy()


def load_development_holdout():
    if DEVELOPMENT_HOLDOUT.exists():
        record = json.loads(DEVELOPMENT_HOLDOUT.read_text(encoding="utf-8"))
    else:
        source = json.loads(LEGACY_SPLIT.read_text(encoding="utf-8"))
        record = {
            "description": (
                "The 41 patients used as the development holdout in experiments inspected "
                "before the corrected five-fold protocol. They are excluded in the "
                "prespecified complementary-162-patient sensitivity analysis."
            ),
            "source_file": str(LEGACY_SPLIT.relative_to(ROOT)),
            "source_seed": source["seed"],
            "patients": source["validation_patients"],
        }
        DEVELOPMENT_HOLDOUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if len(record["patients"]) != 41:
        raise RuntimeError("Development-holdout record must contain 41 patients")
    return set(record["patients"])


def primary_analysis(primary_patients):
    primary = primary_patients[primary_patients.target == PRIMARY_TARGET].copy()
    primary = primary.rename(columns={"mean_scan_dice": "primary_dice"})
    primary = primary[
        [
            "run_id",
            "architecture",
            "cv_fold",
            "patient_id",
            "scan_count",
            "reference_present_scan_count",
            "primary_dice",
        ]
    ].sort_values(["architecture", "patient_id"])
    if primary.primary_dice.isna().any():
        raise RuntimeError("Primary tumor-union Dice contains missing values")

    summaries = []
    for model_index, model in enumerate(MODEL_ORDER):
        values = primary.loc[primary.architecture == model, "primary_dice"].to_numpy()
        ci_low, ci_high = bootstrap_ci(values, "mean", BOOTSTRAP_SEED + model_index)
        summaries.append(
            {
                "architecture": model,
                "model_label": MODEL_LABELS[model],
                "n_patients": len(values),
                "mean_dice": float(np.mean(values)),
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "sd_dice": float(np.std(values, ddof=1)),
                "median_dice": float(np.median(values)),
                "q1_dice": float(np.quantile(values, 0.25)),
                "q3_dice": float(np.quantile(values, 0.75)),
            }
        )
    summary = pd.DataFrame(summaries)

    pivot = primary.pivot(index="patient_id", columns="architecture", values="primary_dice")
    keras_models = MODEL_ORDER[:4]
    friedman = stats.friedmanchisquare(*(pivot[model] for model in keras_models))
    comparisons = []
    for index, comparator in enumerate(keras_models[1:]):
        row = paired_comparison(
            primary, comparator, keras_models[0], seed=BOOTSTRAP_SEED + 100 + index
        )
        row["comparison_family"] = "Controlled Keras; Holm family"
        comparisons.append(row)
    adjusted = holm_adjust([row["p_value"] for row in comparisons])
    for row, value in zip(comparisons, adjusted):
        row["holm_adjusted_p"] = float(value)
        row["gatekeeping_friedman_p"] = float(friedman.pvalue)

    nn_row = paired_comparison(
        primary,
        "nnU-Net v2 3D fullres",
        "Residual 3D U-Net",
        seed=BOOTSTRAP_SEED + 200,
    )
    nn_row["comparison_family"] = "Prespecified whole-pipeline comparison"
    nn_row["holm_adjusted_p"] = math.nan
    nn_row["gatekeeping_friedman_p"] = math.nan
    comparisons.append(nn_row)
    comparison_frame = pd.DataFrame(comparisons)

    friedman_record = pd.DataFrame(
        [
            {
                "test": "Friedman omnibus: four controlled Keras models",
                "n_patients": len(pivot),
                "degrees_of_freedom": 3,
                "statistic": float(friedman.statistic),
                "p_value": float(friedman.pvalue),
            }
        ]
    )
    return primary, summary, comparison_frame, friedman_record


def sensitivity_analysis(primary, excluded_patients):
    sensitivity = primary[~primary.patient_id.isin(excluded_patients)].copy()
    if sensitivity.patient_id.nunique() != 162:
        raise RuntimeError(
            f"Expected 162 complementary patients, found {sensitivity.patient_id.nunique()}"
        )
    summaries = []
    for model_index, model in enumerate(MODEL_ORDER):
        values = sensitivity.loc[sensitivity.architecture == model, "primary_dice"].to_numpy()
        ci_low, ci_high = bootstrap_ci(values, "mean", BOOTSTRAP_SEED + 300 + model_index)
        summaries.append(
            {
                "architecture": model,
                "n_patients": len(values),
                "mean_dice": float(np.mean(values)),
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "median_dice": float(np.median(values)),
            }
        )
    comparisons = []
    for index, comparator in enumerate(MODEL_ORDER[1:4]):
        row = paired_comparison(
            sensitivity,
            comparator,
            MODEL_ORDER[0],
            seed=BOOTSTRAP_SEED + 400 + index,
        )
        comparisons.append(row)
    adjusted = holm_adjust([row["p_value"] for row in comparisons])
    for row, adjusted_p in zip(comparisons, adjusted):
        row["holm_adjusted_p"] = float(adjusted_p)
    nn_row = paired_comparison(
        sensitivity,
        MODEL_ORDER[4],
        MODEL_ORDER[1],
        seed=BOOTSTRAP_SEED + 500,
    )
    nn_row["holm_adjusted_p"] = math.nan
    comparisons.append(nn_row)
    return pd.DataFrame(summaries), pd.DataFrame(comparisons)


def secondary_analysis(primary_patients, primary_cases):
    rows = []
    for model_index, model in enumerate(MODEL_ORDER):
        for target_index, target in enumerate(TARGET_ORDER):
            group = primary_patients[
                (primary_patients.architecture == model)
                & (primary_patients.target == target)
                & (primary_patients.reference_present_scan_count > 0)
            ].copy()
            case_group = primary_cases[
                (primary_cases.architecture == model)
                & (primary_cases.target == target)
                & primary_cases.gt_present
            ]
            dice_values = group.mean_scan_dice.dropna().to_numpy()
            ci_low, ci_high = bootstrap_ci(
                dice_values,
                "mean",
                BOOTSTRAP_SEED + 600 + model_index * len(TARGET_ORDER) + target_index,
            )
            rows.append(
                {
                    "architecture": model,
                    "target": target,
                    "reference_present_patients": int(group.patient_id.nunique()),
                    "reference_present_scans": int(case_group.case_id.nunique()),
                    "mean_patient_dice": float(np.nanmean(group.mean_scan_dice)),
                    "dice_bootstrap_ci95_low": ci_low,
                    "dice_bootstrap_ci95_high": ci_high,
                    "mean_patient_pooled_iou": float(np.nanmean(group.pooled_iou)),
                    "mean_patient_pooled_precision": float(np.nanmean(group.pooled_precision)),
                    "mean_patient_pooled_recall": float(np.nanmean(group.pooled_recall)),
                    "median_patient_hd95_mm": float(np.nanmedian(group.median_scan_hd95_mm)),
                    "mean_patient_average_surface_distance_mm": float(
                        np.nanmean(group.mean_scan_average_surface_distance_mm)
                    ),
                    "mean_patient_surface_dice_1mm": float(
                        np.nanmean(group.mean_scan_surface_dice_1mm)
                    ),
                    "mean_patient_lesion_sensitivity": float(
                        np.nanmean(group.lesion_sensitivity)
                    ),
                    "undefined_surface_scans": int((~case_group.surface_distance_defined).sum()),
                    "surface_defined_scans": int(case_group.surface_distance_defined.sum()),
                }
            )
    return pd.DataFrame(rows)


def absent_reference_analysis(primary_cases):
    rows = []
    absent_targets = [target for target in TARGET_ORDER if "region" not in target.lower()]
    for model in MODEL_ORDER:
        for target in absent_targets:
            group = primary_cases[
                (primary_cases.architecture == model)
                & (primary_cases.target == target)
                & (~primary_cases.gt_present)
            ].copy()
            if group.empty:
                continue
            group["voxel_volume_ml"] = (
                group.voxel_spacing_x_mm
                * group.voxel_spacing_y_mm
                * group.voxel_spacing_z_mm
                / 1000
            )
            group["predicted_volume_ml"] = group.predicted_voxels * group.voxel_volume_ml
            scan_success = int(group.pred_present.sum())
            scan_total = len(group)
            scan_low, scan_high = wilson_ci(scan_success, scan_total)
            patient_any = group.groupby("patient_id").pred_present.any()
            patient_success = int(patient_any.sum())
            patient_total = len(patient_any)
            patient_low, patient_high = wilson_ci(patient_success, patient_total)
            rows.append(
                {
                    "architecture": model,
                    "target": target,
                    "absent_reference_scans": scan_total,
                    "false_positive_scans": scan_success,
                    "false_positive_scan_rate": scan_success / scan_total,
                    "scan_rate_wilson_ci95_low": scan_low,
                    "scan_rate_wilson_ci95_high": scan_high,
                    "patients_with_absent_reference": patient_total,
                    "patients_with_any_false_positive": patient_success,
                    "patient_any_false_positive_rate": patient_success / patient_total,
                    "patient_rate_wilson_ci95_low": patient_low,
                    "patient_rate_wilson_ci95_high": patient_high,
                    "mean_predicted_volume_ml_all_absent_scans": float(
                        group.predicted_volume_ml.mean()
                    ),
                    "median_predicted_volume_ml_all_absent_scans": float(
                        group.predicted_volume_ml.median()
                    ),
                    "p95_predicted_volume_ml_all_absent_scans": float(
                        group.predicted_volume_ml.quantile(0.95)
                    ),
                    "total_false_positive_lesions": int(group.false_positive_lesions.sum()),
                    "median_false_positive_lesions_per_absent_scan": float(
                        group.false_positive_lesions.median()
                    ),
                }
            )
    return pd.DataFrame(rows)


def fold_analysis(primary):
    return (
        primary.groupby(["architecture", "cv_fold"], as_index=False)
        .agg(
            n_patients=("patient_id", "nunique"),
            mean_dice=("primary_dice", "mean"),
            median_dice=("primary_dice", "median"),
            sd_dice=("primary_dice", "std"),
        )
        .sort_values(["architecture", "cv_fold"])
    )


def seed_analysis(patient_data):
    seed_data = patient_data[
        (patient_data.cv_fold == 1) & (patient_data.target == PRIMARY_TARGET)
    ].copy()
    seed_data = seed_data.rename(columns={"mean_scan_dice": "primary_dice"})
    counts = seed_data.groupby(["architecture", "training_seed"]).patient_id.nunique()
    if not (counts == 41).all() or len(counts) != 15:
        raise RuntimeError("Fold-1 seed analysis requires 15 model-seed cells of 41 patients")

    seed_means = (
        seed_data.groupby(["architecture", "training_seed"], as_index=False)
        .agg(n_patients=("patient_id", "nunique"), mean_dice=("primary_dice", "mean"))
    )
    summary_rows = []
    for model in MODEL_ORDER:
        group = seed_data[seed_data.architecture == model]
        seed_pivot = group.pivot(
            index="patient_id", columns="training_seed", values="primary_dice"
        )[[2026, 2027, 2028]]
        means = seed_pivot.mean(axis=0).to_numpy()
        patient_ranges = seed_pivot.max(axis=1) - seed_pivot.min(axis=1)
        patient_sds = seed_pivot.std(axis=1, ddof=1)
        summary_rows.append(
            {
                "architecture": model,
                "fold": 1,
                "patients": len(seed_pivot),
                "seed_2026_mean_dice": float(means[0]),
                "seed_2027_mean_dice": float(means[1]),
                "seed_2028_mean_dice": float(means[2]),
                "mean_across_seed_means": float(means.mean()),
                "sd_across_seed_means": float(means.std(ddof=1)),
                "range_across_seed_means": float(means.max() - means.min()),
                "median_patient_range_across_seeds": float(patient_ranges.median()),
                "median_patient_sd_across_seeds": float(patient_sds.median()),
                "mean_paired_change_seed2027_minus_2026": float(
                    (seed_pivot[2027] - seed_pivot[2026]).mean()
                ),
                "mean_paired_change_seed2028_minus_2026": float(
                    (seed_pivot[2028] - seed_pivot[2026]).mean()
                ),
            }
        )
    return seed_means, pd.DataFrame(summary_rows)


def efficiency_analysis(runs):
    primary_runs = runs[runs.analysis_role == "primary_five_fold_cross_validation"]
    rows = []
    for model in MODEL_ORDER:
        group = primary_runs[primary_runs.architecture == model]
        rows.append(
            {
                "architecture": model,
                "runs": len(group),
                "parameters": int(group.total_parameters.median()),
                "completed_epochs_median": float(group.completed_epochs.median()),
                "completed_epochs_min": int(group.completed_epochs.min()),
                "completed_epochs_max": int(group.completed_epochs.max()),
                "training_hours_median": float(group.training_hours.median()),
                "training_hours_min": float(group.training_hours.min()),
                "training_hours_max": float(group.training_hours.max()),
                "inference_seconds_per_case_median": float(
                    group.inference_seconds_per_case.median()
                ),
                "peak_gpu_memory_gib_median": float(group.peak_gpu_memory_gib.median()),
                "peak_gpu_memory_gib_min": float(group.peak_gpu_memory_gib.min()),
                "peak_gpu_memory_gib_max": float(group.peak_gpu_memory_gib.max()),
                "archive_mib_median": float(group.archive_mib.median()),
                "platforms": "; ".join(sorted(group.platform.unique())),
            }
        )
    return pd.DataFrame(rows)


def style_axis(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#d9dee3", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)


def figure_primary(primary, summary, comparisons):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.5), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    data = [primary.loc[primary.architecture == model, "primary_dice"] for model in MODEL_ORDER]
    violin = ax.violinplot(data, positions=np.arange(1, 6), widths=0.78, showextrema=False)
    for body, model in zip(violin["bodies"], MODEL_ORDER):
        body.set_facecolor(MODEL_COLORS[model])
        body.set_edgecolor("white")
        body.set_alpha(0.32)
    box = ax.boxplot(
        data,
        positions=np.arange(1, 6),
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.4},
        whiskerprops={"color": "#444444"},
        capprops={"color": "#444444"},
    )
    for patch, model in zip(box["boxes"], MODEL_ORDER):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_alpha(0.72)
    for x, model in enumerate(MODEL_ORDER, start=1):
        row = summary[summary.architecture == model].iloc[0]
        ax.errorbar(
            x,
            row.mean_dice,
            yerr=[[row.mean_dice - row.bootstrap_ci95_low], [row.bootstrap_ci95_high - row.mean_dice]],
            fmt="o",
            color="#101820",
            markersize=4.5,
            capsize=3,
            linewidth=1.2,
            zorder=5,
        )
    ax.set_xticks(np.arange(1, 6), [MODEL_LABELS[m] for m in MODEL_ORDER], rotation=22, ha="right")
    ax.set_ylabel("Patient-level tumor-union Dice")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("A  Out-of-fold distributions (n = 203 patients)", loc="left", weight="bold")
    style_axis(ax)

    ax = axes[1]
    display = comparisons.iloc[[0, 1, 2, 3]].copy()
    ypos = np.arange(len(display))[::-1]
    for y, (_, row) in zip(ypos, display.iterrows()):
        color = MODEL_COLORS[row.comparator]
        ax.errorbar(
            row.median_paired_difference,
            y,
            xerr=[
                [row.median_paired_difference - row.bootstrap_ci95_low],
                [row.bootstrap_ci95_high - row.median_paired_difference],
            ],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=6,
        )
    ax.axvline(0, color="#333333", linewidth=1, linestyle="--")
    ax.set_yticks(ypos, display.comparison.str.replace(" vs ", "\nvs ", regex=False))
    ax.set_xlabel("Median paired Dice difference (95% bootstrap CI)")
    ax.set_title("B  Prespecified paired effects", loc="left", weight="bold")
    ax.grid(axis="x", color="#d9dee3", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "primary_performance.png", dpi=320, bbox_inches="tight")
    fig.savefig(FIGURES / "primary_performance.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_fold_seed(folds, seed_means):
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.15))
    ax = axes[0]
    for model in MODEL_ORDER:
        group = folds[folds.architecture == model].sort_values("cv_fold")
        ax.plot(
            group.cv_fold,
            group.mean_dice,
            marker="o",
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
            linewidth=1.7,
        )
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Outer fold")
    ax.set_ylabel("Mean patient-level tumor-union Dice")
    ax.set_ylim(0.68, 0.95)
    ax.set_title("A  Fold variation (seed 2026)", loc="left", weight="bold")
    style_axis(ax)

    ax = axes[1]
    for model in MODEL_ORDER:
        group = seed_means[seed_means.architecture == model].sort_values("training_seed")
        ax.plot(
            group.training_seed.astype(str),
            group.mean_dice,
            marker="o",
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
            linewidth=1.7,
        )
    ax.set_xlabel("Training seed (fold 1 only)")
    ax.set_ylabel("Mean patient-level tumor-union Dice")
    ax.set_ylim(0.68, 0.95)
    ax.set_title("B  Training-seed sensitivity", loc="left", weight="bold")
    style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIGURES / "fold_seed_sensitivity.png", dpi=320, bbox_inches="tight")
    fig.savefig(FIGURES / "fold_seed_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_target_metrics(secondary):
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.9))
    x = np.arange(len(TARGET_ORDER))
    offsets = np.linspace(-0.28, 0.28, len(MODEL_ORDER))
    for offset, model in zip(offsets, MODEL_ORDER):
        group = secondary[secondary.architecture == model].set_index("target").loc[TARGET_ORDER]
        axes[0].errorbar(
            x + offset,
            group.mean_patient_dice,
            yerr=[
                group.mean_patient_dice - group.dice_bootstrap_ci95_low,
                group.dice_bootstrap_ci95_high - group.mean_patient_dice,
            ],
            fmt="o",
            capsize=2,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
            markersize=4.5,
        )
        axes[1].plot(
            x + offset,
            group.median_patient_hd95_mm,
            "o",
            color=MODEL_COLORS[model],
            markersize=4.5,
        )
    axes[0].set_ylabel("Mean patient-level Dice (95% bootstrap CI)")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("A  Reference-present overlap", loc="left", weight="bold")
    axes[1].set_ylabel("Median patient HD95 (mm; lower is better)")
    axes[1].set_yscale("log")
    axes[1].set_title("B  Reference-present boundary error", loc="left", weight="bold")
    for ax in axes:
        ax.set_xticks(x, [TARGET_SHORT[t] for t in TARGET_ORDER], rotation=28, ha="right")
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(FIGURES / "target_metrics.png", dpi=320, bbox_inches="tight")
    fig.savefig(FIGURES / "target_metrics.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_absent_reference(absent):
    targets = [
        "Non-enhancing tumor core",
        "Enhancing tissue",
        "Resection cavity",
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(targets))
    offsets = np.linspace(-0.28, 0.28, len(MODEL_ORDER))
    for offset, model in zip(offsets, MODEL_ORDER):
        group = absent[absent.architecture == model].set_index("target").loc[targets]
        rate = group.false_positive_scan_rate.to_numpy()
        lower_error = np.maximum(
            0.0, rate - group.scan_rate_wilson_ci95_low.to_numpy()
        )
        upper_error = np.maximum(
            0.0, group.scan_rate_wilson_ci95_high.to_numpy() - rate
        )
        ax.errorbar(
            x + offset,
            rate,
            yerr=[lower_error, upper_error],
            fmt="o",
            capsize=2,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
            markersize=5,
        )
    denominators = (
        absent[absent.architecture == MODEL_ORDER[0]].set_index("target").loc[targets]
    )
    labels = [
        f"{TARGET_SHORT[target]}\n(n = {int(denominators.loc[target, 'absent_reference_scans'])} scans)"
        for target in targets
    ]
    ax.set_xticks(x, labels)
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("False-positive scan rate (Wilson 95% CI)")
    ax.set_title("Predictions when the reference target is absent", loc="left", weight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=5, frameon=False)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "absent_reference_false_positives.png", dpi=320, bbox_inches="tight")
    fig.savefig(FIGURES / "absent_reference_false_positives.pdf", bbox_inches="tight")
    plt.close(fig)


def select_qualitative_cases(primary_cases):
    tumor = primary_cases[primary_cases.target == PRIMARY_TARGET]
    pivot = tumor.pivot(index="case_id", columns="architecture", values="dice")[MODEL_ORDER]
    scores = pd.DataFrame(
        {
            "cross_model_mean_dice": pivot.mean(axis=1),
            "cross_model_sd_dice": pivot.std(axis=1, ddof=1),
            "cross_model_min_dice": pivot.min(axis=1),
            "cross_model_max_dice": pivot.max(axis=1),
        }
    )
    scores["patient_id"] = scores.index.to_series().str.extract(r"^(PatientID_\d+)")[0]
    scores["fold"] = scores.index.to_series().map(
        tumor.drop_duplicates("case_id").set_index("case_id").cv_fold
    )
    chosen = []
    used_patients = set()

    def choose(label, ordering):
        for case_id in ordering:
            patient = scores.loc[case_id, "patient_id"]
            if patient not in used_patients:
                used_patients.add(patient)
                row = scores.loc[case_id].to_dict()
                row.update({"selection_label": label, "case_id": case_id})
                chosen.append(row)
                return
        raise RuntimeError(f"Could not select a distinct-patient case for {label}")

    median_value = scores.cross_model_mean_dice.median()
    typical_order = (scores.cross_model_mean_dice - median_value).abs().sort_values().index
    choose("Median-typical", typical_order)
    q10 = scores.cross_model_mean_dice.quantile(0.10)
    challenging_order = (scores.cross_model_mean_dice - q10).abs().sort_values().index
    choose("Lower-decile challenge", challenging_order)
    disagreement_order = scores.cross_model_sd_dice.sort_values(ascending=False).index
    choose("Largest model disagreement", disagreement_order)
    return pd.DataFrame(chosen)


def extract_prediction(zpath: Path, case_id: str, temp_dir: Path):
    with zipfile.ZipFile(zpath) as archive:
        matches = [
            name
            for name in archive.namelist()
            if name.endswith(f"outer_test_predictions/{case_id}.nii.gz")
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Prediction for {case_id} not uniquely found in {zpath}")
        destination = temp_dir / f"{zpath.parent.name}_{case_id}.nii.gz"
        destination.write_bytes(archive.read(matches[0]))
    return np.asarray(nib.load(destination).dataobj)


def case_source_path(case_id: str, suffix: str):
    patient, timepoint_number = case_id.split("_Timepoint_")
    timepoint = f"Timepoint_{timepoint_number}"
    path = DATASET_ROOT / patient / timepoint / f"{case_id}_{suffix}.nii.gz"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def orient(array):
    return np.rot90(array)


def crop_bounds(mask, margin=16, minimum=84):
    ys, xs = np.where(mask)
    if not len(xs):
        return 0, mask.shape[1], 0, mask.shape[0]
    x0, x1 = max(0, xs.min() - margin), min(mask.shape[1], xs.max() + margin + 1)
    y0, y1 = max(0, ys.min() - margin), min(mask.shape[0], ys.max() + margin + 1)
    if x1 - x0 < minimum:
        extra = minimum - (x1 - x0)
        x0 = max(0, x0 - extra // 2)
        x1 = min(mask.shape[1], x1 + extra - extra // 2)
    if y1 - y0 < minimum:
        extra = minimum - (y1 - y0)
        y0 = max(0, y0 - extra // 2)
        y1 = min(mask.shape[0], y1 + extra - extra // 2)
    return x0, x1, y0, y1


def figure_qualitative(selection, primary_cases):
    run_lookup = (
        primary_cases[["architecture", "cv_fold", "run_id"]]
        .drop_duplicates()
        .set_index(["architecture", "cv_fold"])
        .run_id
    )
    mask_cmap = ListedColormap(["#00000000", "#f94144", "#43aa8b", "#f9c74f", "#577590"])
    column_labels = ["T1c + reference"] + [MODEL_LABELS[m] for m in MODEL_ORDER]
    fig, axes = plt.subplots(len(selection), len(column_labels), figsize=(13.4, 7.4))
    with tempfile.TemporaryDirectory(prefix="mu_glioma_qualitative_") as directory:
        temp_dir = Path(directory)
        for row_index, row in selection.reset_index(drop=True).iterrows():
            case_id = row.case_id
            fold = int(row.fold)
            image = np.asarray(nib.load(case_source_path(case_id, "brain_t1c")).dataobj)
            truth = np.asarray(nib.load(case_source_path(case_id, "tumorMask")).dataobj)
            tumor = np.isin(truth, [1, 2, 3])
            areas = tumor.sum(axis=(0, 1))
            slice_index = int(np.argmax(areas))
            image_slice = orient(image[:, :, slice_index])
            truth_slice = orient(truth[:, :, slice_index])
            oriented_tumor = orient(tumor[:, :, slice_index])
            x0, x1, y0, y1 = crop_bounds(oriented_tumor)
            nonzero = image_slice[image_slice > 0]
            lower, upper = (
                np.percentile(nonzero, [1, 99.5]) if nonzero.size else (image_slice.min(), image_slice.max())
            )
            masks = [truth_slice]
            for model in MODEL_ORDER:
                run_id = run_lookup.loc[(model, fold)]
                prediction = extract_prediction(zip_path(run_id), case_id, temp_dir)
                masks.append(orient(prediction[:, :, slice_index]))

            for column_index, (column, mask) in enumerate(zip(column_labels, masks)):
                ax = axes[row_index, column_index]
                ax.imshow(image_slice[y0:y1, x0:x1], cmap="gray", vmin=lower, vmax=upper)
                masked = np.ma.masked_where(mask[y0:y1, x0:x1] == 0, mask[y0:y1, x0:x1])
                ax.imshow(masked, cmap=mask_cmap, vmin=0, vmax=4, alpha=0.58, interpolation="nearest")
                ax.axis("off")
                if row_index == 0:
                    ax.set_title(column, fontsize=9.2, weight="bold")
            axes[row_index, 0].text(
                -0.07,
                0.5,
                (
                    f"{row.selection_label}\n{case_id.replace('PatientID_', 'P').replace('_Timepoint_', ' / T')}\n"
                    f"cross-model mean {row.cross_model_mean_dice:.2f}"
                ),
                transform=axes[row_index, 0].transAxes,
                ha="right",
                va="center",
                fontsize=8.2,
            )
    legend = [
        Patch(facecolor="#f94144", label="Non-enhancing core"),
        Patch(facecolor="#43aa8b", label="FLAIR/edema"),
        Patch(facecolor="#f9c74f", label="Enhancing tissue"),
        Patch(facecolor="#577590", label="Resection cavity"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.56, 0.01))
    fig.subplots_adjust(left=0.15, right=0.995, top=0.94, bottom=0.08, wspace=0.025, hspace=0.06)
    fig.savefig(FIGURES / "qualitative_predictions.png", dpi=320, bbox_inches="tight")
    fig.savefig(FIGURES / "qualitative_predictions.pdf", bbox_inches="tight")
    plt.close(fig)


def write_machine_summary(
    primary_summary,
    comparisons,
    friedman,
    sensitivity_summary,
    sensitivity_comparisons,
    secondary,
    absent,
    folds,
    seed_summary,
    efficiency,
):
    def records(frame):
        return json.loads(frame.to_json(orient="records"))

    summary = {
        "analysis_version": "final_35_run_prespecified_v1",
        "primary_target": PRIMARY_TARGET,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "primary_summary": records(primary_summary),
        "friedman": records(friedman)[0],
        "paired_comparisons": records(comparisons),
        "complementary_162_patient_summary": records(sensitivity_summary),
        "complementary_162_patient_comparisons": records(sensitivity_comparisons),
        "secondary_summary": records(secondary),
        "absent_reference_summary": records(absent),
        "fold_summary": records(folds),
        "seed_summary": records(seed_summary),
        "efficiency_summary": records(efficiency),
    }
    (OUTPUT / "paper_numbers.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_manifest(runs):
    record = {
        "analysis_version": "final_35_run_prespecified_v1",
        "analysis_freeze_date": "2026-09-04",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__)),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "nibabel": nib.__version__,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "validated_archives": runs[["run_id", "zip_path", "zip_sha256"]].to_dict("records"),
    }
    (OUTPUT / "analysis_manifest.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )


def main():
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    tracker, patient_data, case_data, runs = collect_data()
    primary_patients, primary_cases = validate_primary(patient_data, case_data)
    development_holdout = load_development_holdout()

    primary, primary_summary, comparisons, friedman = primary_analysis(primary_patients)
    sensitivity_summary, sensitivity_comparisons = sensitivity_analysis(
        primary, development_holdout
    )
    secondary = secondary_analysis(primary_patients, primary_cases)
    absent = absent_reference_analysis(primary_cases)
    folds = fold_analysis(primary)
    seed_means, seed_summary = seed_analysis(patient_data)
    efficiency = efficiency_analysis(runs)
    qualitative = select_qualitative_cases(primary_cases)

    outputs = {
        "primary_patient_metrics.csv": primary,
        "primary_model_summary.csv": primary_summary,
        "primary_paired_comparisons.csv": comparisons,
        "primary_friedman_test.csv": friedman,
        "complementary_162_patient_summary.csv": sensitivity_summary,
        "complementary_162_patient_comparisons.csv": sensitivity_comparisons,
        "target_metrics_summary.csv": secondary,
        "absent_reference_summary.csv": absent,
        "fold_summary.csv": folds,
        "seed_fold1_means.csv": seed_means,
        "seed_sensitivity_summary.csv": seed_summary,
        "run_efficiency.csv": runs,
        "efficiency_summary.csv": efficiency,
        "qualitative_case_selection.csv": qualitative,
    }
    for filename, frame in outputs.items():
        frame.to_csv(OUTPUT / filename, index=False)

    figure_primary(primary, primary_summary, comparisons)
    figure_fold_seed(folds, seed_means)
    figure_target_metrics(secondary)
    figure_absent_reference(absent)
    figure_qualitative(qualitative, primary_cases)
    write_machine_summary(
        primary_summary,
        comparisons,
        friedman,
        sensitivity_summary,
        sensitivity_comparisons,
        secondary,
        absent,
        folds,
        seed_summary,
        efficiency,
    )
    write_manifest(runs)
    print(f"Final analysis complete: {OUTPUT}")
    print(primary_summary.to_string(index=False))
    print(friedman.to_string(index=False))
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()

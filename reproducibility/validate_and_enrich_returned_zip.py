#!/usr/bin/env python3
"""Enrich and validate returned MU-Glioma result ZIPs without retraining.

The original archive is never overwritten. A sibling ``*_DETAILED.zip`` is
created after recomputing all metrics from the archived prediction NIfTIs and
the authoritative local masks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import tempfile
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from mu_glioma_detailed_metrics import (
    METRIC_DEFINITIONS,
    REQUIRED_PER_CASE_COLUMNS,
    SURFACE_DICE_TOLERANCE_MM,
    build_prediction_manifest,
    build_uniform_summaries,
    evaluate_case_targets,
    sha256_file,
    validate_discrete_label_volume,
)


CONTRACT_VERSION = "mu_glioma_detailed_zip_v2_nested"


def index_truth_masks(data_root: Path) -> dict[str, dict]:
    cases = {}
    modalities = ("brain_t1n", "brain_t1c", "brain_t2w", "brain_t2f")
    for patient_dir in sorted(data_root.glob("PatientID_*")):
        for timepoint_dir in sorted(patient_dir.glob("Timepoint_*")):
            case_id = f"{patient_dir.name}_{timepoint_dir.name}"
            mask = timepoint_dir / f"{case_id}_tumorMask.nii.gz"
            images = [
                timepoint_dir / f"{case_id}_{modality}.nii.gz"
                for modality in modalities
            ]
            if mask.exists() and all(path.exists() for path in images):
                cases[case_id] = {
                    "id": case_id,
                    "patient": patient_dir.name,
                    "mask": mask,
                }
    assert len(cases) == 594, f"Expected 594 labelled local cases; found {len(cases)}"
    return cases


def read_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 35, f"Expected 35 manifest rows; found {len(rows)}"
    return rows


def manifest_row_for_zip(zip_path: Path, rows: list[dict]) -> dict:
    exact = [row for row in rows if row["expected_zip"] == zip_path.name]
    if len(exact) == 1:
        return exact[0]
    candidates = [
        row
        for row in rows
        if zip_path.stem.startswith(Path(row["expected_zip"]).stem)
    ]
    assert len(candidates) == 1, (
        f"Could not map {zip_path.name} to exactly one manifest row: "
        f"{[row['run_id'] for row in candidates]}"
    )
    return candidates[0]


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        assert target == destination or destination in target.parents, (
            f"Unsafe archive member: {member.filename}"
        )
    archive.extractall(destination)


def locate_unique(root: Path, name: str) -> Path:
    matches = [path for path in root.rglob(name) if path.is_file()]
    assert len(matches) == 1, (
        f"Archive must contain exactly one {name}; found "
        f"{[str(path.relative_to(root)) for path in matches]}"
    )
    return matches[0]


def locate_prediction_dir(root: Path) -> Path:
    candidates = [
        path
        for name in ("outer_test_predictions", "validation_predictions")
        for path in root.rglob(name)
        if path.is_dir() and list(path.glob("*.nii.gz"))
    ]
    assert len(candidates) == 1, (
        f"Expected one validation_predictions directory; found {candidates}"
    )
    return candidates[0]


def report_directory(extracted_root: Path, split_path: Path) -> Path:
    reports = extracted_root / "reports"
    if reports.exists():
        return reports
    return split_path.parent


def validate_source_artifacts(extracted_root: Path) -> dict:
    """Reject incomplete returns before adding any derived reports."""
    files = [path for path in extracted_root.rglob("*") if path.is_file()]
    basenames = {path.name for path in files}
    configs = [
        path
        for path in files
        if path.name in {"model_config.json", "experiment_config.json"}
    ]
    checkpoints = [
        path
        for path in files
        if path.suffix.lower() in {".keras", ".h5", ".hdf5", ".pth", ".pt"}
    ]
    training_logs = [
        path
        for path in files
        if path.name == "training_log.csv"
        or (path.name.startswith("training_log_") and path.suffix == ".txt")
    ]
    convergence_plots = [
        path
        for path in files
        if path.name in {"learning_curves.png", "progress.png"}
    ]
    assert len(configs) == 1, (
        "Archive must contain exactly one model/experiment configuration; "
        f"found {[str(path.relative_to(extracted_root)) for path in configs]}"
    )
    assert checkpoints, "Archive lacks a best model/checkpoint"
    assert training_logs, "Archive lacks a complete training log"
    assert convergence_plots, "Archive lacks learning_curves.png or progress.png"
    assert all(path.stat().st_size > 0 for path in checkpoints), "Empty checkpoint"
    assert all(path.stat().st_size > 0 for path in training_logs), "Empty training log"
    assert all(path.stat().st_size > 0 for path in convergence_plots), (
        "Empty convergence plot"
    )
    if "outer_test_predictions" in {path.name for path in extracted_root.rglob("*")}:
        assert "efficiency.json" in basenames, "Corrected archive lacks efficiency.json"
        assert "environment.json" in basenames, "Corrected archive lacks environment.json"
        efficiency_path = next(
            path for path in files if path.name == "efficiency.json"
        )
        efficiency = json.loads(efficiency_path.read_text(encoding="utf-8"))
        assert int(efficiency.get("total_parameters", 0)) > 0, (
            "Efficiency record lacks total parameter count"
        )
        assert any(
            float(efficiency.get(key, 0) or 0) > 0
            for key in (
                "training_wall_seconds",
                "training_and_tuning_wall_seconds",
            )
        ), "Efficiency record lacks training wall time"
        assert float(
            efficiency.get(
                "outer_test_inference_total_wall_seconds_including_io",
                efficiency.get("outer_test_inference_wall_seconds", 0),
            )
            or 0
        ) > 0, "Efficiency record lacks outer-test inference time"
        assert any(
            key in efficiency
            for key in ("peak_gpu_memory_bytes", "peak_observed_gpu_memory_mib")
        ), "Efficiency record lacks peak GPU-memory measurement"
    config_record = json.loads(configs[0].read_text(encoding="utf-8"))
    assert isinstance(config_record, dict), "Model/experiment configuration is not JSON"
    if configs[0].name == "model_config.json":
        assert "fit_complete.json" in basenames, (
            "Keras archive lacks the clean model.fit completion marker"
        )
        completion_path = next(path for path in files if path.name == "fit_complete.json")
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        assert completion.get("status") == "fit_complete"
        assert completion.get("run_id") == config_record.get("run_id")
    else:
        assert "preflight_passed.json" in basenames, (
            "nnU-Net archive lacks the real-data/GPU preflight success record"
        )
        preflight_path = next(
            path for path in files if path.name == "preflight_passed.json"
        )
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        assert preflight.get("status") == "passed"
        assert preflight.get("real_fit_train_batch") is True
        assert preflight.get("real_inner_tuning_batch") is True
    return {
        "source_files": len(files),
        "configuration_files": [
            str(path.relative_to(extracted_root)) for path in configs
        ],
        "model_or_checkpoint_files": [
            str(path.relative_to(extracted_root)) for path in checkpoints
        ],
        "training_log_files": [
            str(path.relative_to(extracted_root)) for path in training_logs
        ],
        "convergence_plot_files": [
            str(path.relative_to(extracted_root)) for path in convergence_plots
        ],
        "training_environment_record_present": "environment.json" in basenames,
        "recorded_run_id": config_record.get("run_id"),
        "recorded_cv_fold": config_record.get("cv_fold"),
        "recorded_training_seed": config_record.get(
            "training_seed", config_record.get("seed")
        ),
    }


def evaluate_archive(
    extracted_root: Path,
    row: dict,
    truth_cases: dict[str, dict],
    source_zip: Path,
) -> Path:
    source_inventory = validate_source_artifacts(extracted_root)
    split_path = locate_unique(extracted_root, "patient_split.json")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split_text = json.dumps(split, indent=2)
    split_hash = hashlib.sha256(split_text.encode("utf-8")).hexdigest()
    assert split_hash == row["split_sha256"], (
        f"Split mismatch for {row['run_id']}: {split_hash} != {row['split_sha256']}"
    )
    nested_design = "outer_test_cases" in split
    evaluation_case_ids = (
        split["outer_test_cases"] if nested_design else split["validation_cases"]
    )
    expected_case_count = int(
        row.get("outer_test_cases") or row.get("validation_cases")
    )
    assert len(evaluation_case_ids) == expected_case_count
    assert all(case_id in truth_cases for case_id in evaluation_case_ids)
    if nested_design:
        fit_patients = set(split["fit_train_patients"])
        tuning_patients = set(split["tuning_patients"])
        evaluation_patients = set(split["outer_test_patients"])
        assert fit_patients.isdisjoint(tuning_patients)
        assert fit_patients.isdisjoint(evaluation_patients)
        assert tuning_patients.isdisjoint(evaluation_patients)
        assert fit_patients | tuning_patients | evaluation_patients == set(truth["patient"] for truth in truth_cases.values())
        fit_cases = set(split["fit_train_cases"])
        tuning_cases = set(split["tuning_cases"])
        evaluation_cases = set(split["outer_test_cases"])
        assert fit_cases.isdisjoint(tuning_cases)
        assert fit_cases.isdisjoint(evaluation_cases)
        assert tuning_cases.isdisjoint(evaluation_cases)
        assert fit_cases | tuning_cases | evaluation_cases == set(truth_cases)
        assert {
            truth_cases[case_id]["patient"] for case_id in fit_cases
        } == fit_patients
        assert {
            truth_cases[case_id]["patient"] for case_id in tuning_cases
        } == tuning_patients
        assert len(fit_cases) == int(row["fit_train_cases"])
        assert len(tuning_cases) == int(row["tuning_cases"])
        assert len(fit_patients) == int(row["fit_train_patients"])
        assert len(tuning_patients) == int(row["tuning_patients"])
        assert len(evaluation_patients) == int(row["outer_test_patients"])
    else:
        assert not (
            set(split["train_patients"]) & set(split["validation_patients"])
        ), "Patient leakage in archived split"
        evaluation_patients = set(split["validation_patients"])
    derived_evaluation_patients = {
        truth_cases[case_id]["patient"] for case_id in evaluation_case_ids
    }
    assert derived_evaluation_patients == evaluation_patients, (
        "Archived outer-test patient list does not match its cases"
    )

    config_path = next(
        path
        for path in extracted_root.rglob("*")
        if path.is_file()
        and path.name in {"model_config.json", "experiment_config.json"}
    )
    config_record = json.loads(config_path.read_text(encoding="utf-8"))
    recorded_hash = config_record.get(
        "split_sha256", config_record.get("expected_split_sha256")
    )
    if recorded_hash is not None:
        assert recorded_hash == split_hash, "Configuration split checksum mismatch"
    if nested_design:
        assert config_record.get("archive_contract_version") == CONTRACT_VERSION
        assert config_record.get("checkpoint_selection_role") == (
            "patient_grouped_inner_tuning"
        )
        assert config_record.get("final_evaluation_role") == (
            "untouched_outer_internal_test"
        )
    identity_checks = {
        "run_id": row["run_id"],
        "cv_fold": int(row["cv_fold"]),
        "split_seed": int(row.get("outer_split_seed") or row.get("split_seed")),
        "inner_split_seed": int(row.get("inner_split_seed") or 0),
        "training_seed": int(row["training_seed"]),
    }
    for key, expected in identity_checks.items():
        recorded = config_record.get(key)
        if key == "inner_split_seed" and not nested_design:
            continue
        if key == "training_seed" and recorded is None:
            recorded = config_record.get("seed")
        if recorded is not None:
            assert recorded == expected, (
                f"Configuration {key} mismatch: {recorded} != {expected}"
            )

    prediction_dir = locate_prediction_dir(extracted_root)
    predictions = {
        path.name.removesuffix(".nii.gz"): path
        for path in prediction_dir.glob("*.nii.gz")
    }
    assert set(predictions) == set(evaluation_case_ids), (
        f"Prediction set mismatch: missing={sorted(set(evaluation_case_ids) - set(predictions))[:10]}, "
        f"extra={sorted(set(predictions) - set(evaluation_case_ids))[:10]}"
    )

    rows = []
    for number, case_id in enumerate(evaluation_case_ids, 1):
        truth_case = truth_cases[case_id]
        truth_image = nib.load(truth_case["mask"])
        prediction_image = nib.load(predictions[case_id])
        truth_labels = validate_discrete_label_volume(
            np.asanyarray(truth_image.dataobj), f"{case_id} reference"
        )
        prediction_labels = validate_discrete_label_volume(
            np.asanyarray(prediction_image.dataobj), f"{case_id} prediction"
        )
        assert truth_labels.shape == prediction_labels.shape, case_id
        assert np.allclose(truth_image.affine, prediction_image.affine, atol=1e-4), case_id
        spacing_mm = tuple(float(value) for value in truth_image.header.get_zooms()[:3])
        rows.extend(
            evaluate_case_targets(
                truth_labels,
                prediction_labels,
                spacing_mm,
                {
                    "run_id": row["run_id"],
                    "model": row["architecture"],
                    "cv_fold": int(row["cv_fold"]),
                    "split_seed": int(
                        row.get("outer_split_seed") or row.get("split_seed")
                    ),
                    "training_seed": int(row["training_seed"]),
                    "case_id": case_id,
                    "patient_id": truth_case["patient"],
                },
            )
        )
        if number % 10 == 0 or number == len(evaluation_case_ids):
            print(f"{row['run_id']}: evaluated {number}/{len(evaluation_case_ids)} predictions")

    scores = pd.DataFrame(rows)
    assert list(scores.columns) == REQUIRED_PER_CASE_COLUMNS
    patient_scores, summary, absent_summary = build_uniform_summaries(scores)
    reports = report_directory(extracted_root, split_path)
    reports.mkdir(parents=True, exist_ok=True)
    per_case_name = (
        "outer_test_per_case_metrics.csv"
        if nested_design
        else "validation_per_case_metrics.csv"
    )
    per_patient_name = (
        "outer_test_per_patient_metrics.csv"
        if nested_design
        else "validation_per_patient_metrics.csv"
    )
    prediction_manifest_name = (
        "outer_test_prediction_manifest.csv"
        if nested_design
        else "validation_prediction_manifest.csv"
    )
    scores.to_csv(reports / per_case_name, index=False)
    patient_scores.to_csv(reports / per_patient_name, index=False)
    summary.to_csv(reports / "final_summary.csv", index=False)
    absent_summary.to_csv(
        reports / "absent_reference_false_positive_summary.csv", index=False
    )
    prediction_manifest = build_prediction_manifest(
        prediction_dir,
        [truth_cases[case_id] for case_id in evaluation_case_ids],
    )
    prediction_manifest.to_csv(
        reports / prediction_manifest_name, index=False
    )
    (reports / "metric_definitions.json").write_text(
        json.dumps(METRIC_DEFINITIONS, indent=2), encoding="utf-8"
    )
    provenance = {
        "archive_contract_version": CONTRACT_VERSION,
        "run_id": row["run_id"],
        "source_archive": source_zip.name,
        "source_archive_bytes": source_zip.stat().st_size,
        "source_archive_sha256": sha256_file(source_zip),
        "split_sha256": split_hash,
        "prediction_files": len(prediction_manifest),
        "per_case_metric_rows": len(scores),
        "surface_dice_tolerance_mm": SURFACE_DICE_TOLERANCE_MM,
        "enriched_with_local_ground_truth": True,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "nibabel": nib.__version__,
        "scipy": importlib.metadata.version("scipy"),
        "surface_distance": importlib.metadata.version("surface-distance"),
    }
    (reports / "enrichment_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    (reports / "source_artifact_inventory.json").write_text(
        json.dumps(source_inventory, indent=2), encoding="utf-8"
    )
    artifact_manifest = {
        **provenance,
        "includes_all_prediction_volumes": True,
        "includes_surface_metrics": True,
        "includes_lesion_metrics": True,
        "includes_absent_reference_uncertainty": True,
        "required_per_case_columns_present": True,
        "source_artifacts_validated": True,
        "source_inventory": source_inventory,
    }
    (reports / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2), encoding="utf-8"
    )

    checksum_path = reports / "artifact_checksums.csv"
    if checksum_path.exists():
        checksum_path.unlink()
    checksum_rows = []
    for path in sorted(extracted_root.rglob("*")):
        if path.is_file():
            checksum_rows.append(
                {
                    "archive_path": str(path.relative_to(extracted_root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    pd.DataFrame(checksum_rows).to_csv(checksum_path, index=False)
    return reports


def validate_detailed_zip(
    zip_path: Path, expected_predictions: int, nested_design: bool
) -> None:
    required_basenames = {
        "patient_split.json",
        "metric_definitions.json",
        "absent_reference_false_positive_summary.csv",
        "final_summary.csv",
        "artifact_manifest.json",
        "artifact_checksums.csv",
        "enrichment_provenance.json",
        "source_artifact_inventory.json",
    }
    required_basenames |= (
        {
            "outer_test_per_case_metrics.csv",
            "outer_test_per_patient_metrics.csv",
            "outer_test_prediction_manifest.csv",
            "efficiency.json",
            "environment.json",
        }
        if nested_design
        else {
            "validation_per_case_metrics.csv",
            "validation_per_patient_metrics.csv",
            "validation_prediction_manifest.csv",
        }
    )
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)), "Duplicate paths in detailed archive"
        basenames = {Path(name).name for name in names}
        missing = required_basenames - basenames
        assert not missing, f"Detailed archive missing: {sorted(missing)}"
        predictions = [
            name
            for name in names
            if (
                "outer_test_predictions/" in name
                or "validation_predictions/" in name
            )
            and name.endswith(".nii.gz")
        ]
        assert len(predictions) == expected_predictions, (
            f"Expected {expected_predictions} predictions; archived {len(predictions)}"
        )
        metrics_name = next(
            name
            for name in names
            if Path(name).name
            == (
                "outer_test_per_case_metrics.csv"
                if nested_design
                else "validation_per_case_metrics.csv"
            )
        )
        with archive.open(metrics_name) as handle:
            columns = pd.read_csv(handle, nrows=0).columns.tolist()
        missing_columns = [
            column for column in REQUIRED_PER_CASE_COLUMNS if column not in columns
        ]
        assert not missing_columns, f"Missing metric columns: {missing_columns}"

        checksum_name = next(
            name for name in names if Path(name).name == "artifact_checksums.csv"
        )
        with archive.open(checksum_name) as handle:
            checksum_table = pd.read_csv(handle)
        assert set(checksum_table.columns) == {"archive_path", "bytes", "sha256"}
        indexed_paths = set(checksum_table.archive_path)
        expected_indexed_paths = set(names) - {checksum_name}
        assert indexed_paths == expected_indexed_paths, (
            "Artifact checksum index does not cover every non-index file: "
            f"missing={sorted(expected_indexed_paths - indexed_paths)[:10]}, "
            f"extra={sorted(indexed_paths - expected_indexed_paths)[:10]}"
        )
        for record in checksum_table.itertuples(index=False):
            digest = hashlib.sha256()
            byte_count = 0
            with archive.open(record.archive_path) as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    byte_count += len(chunk)
                    digest.update(chunk)
            assert byte_count == int(record.bytes), record.archive_path
            assert digest.hexdigest() == record.sha256, record.archive_path


def enrich_one(
    zip_path: Path,
    row: dict,
    truth_cases: dict[str, dict],
    output_dir: Path | None,
) -> Path:
    destination = (
        (output_dir / f"{zip_path.stem}_DETAILED.zip")
        if output_dir is not None
        else zip_path.with_name(f"{zip_path.stem}_DETAILED.zip")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mu_glioma_zip_") as temp:
        extracted = Path(temp) / "archive"
        extracted.mkdir()
        with zipfile.ZipFile(zip_path) as archive:
            safe_extract(archive, extracted)
        evaluate_archive(extracted, row, truth_cases, zip_path)
        split_path = locate_unique(extracted, "patient_split.json")
        split = json.loads(split_path.read_text(encoding="utf-8"))
        nested_design = "outer_test_cases" in split
        with zipfile.ZipFile(
            destination, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
        ) as archive:
            for path in sorted(extracted.rglob("*")):
                if path.is_file():
                    archive.write(path, arcname=str(path.relative_to(extracted)))
    expected_predictions = int(
        row.get("outer_test_cases") or row.get("validation_cases")
    )
    validate_detailed_zip(destination, expected_predictions, nested_design)
    print("Validated detailed archive:", destination)
    return destination


def default_data_root(script_path: Path) -> Path:
    candidates = [
        script_path.parent / "MU-Glioma-Post",
        script_path.parent.parent / "MU-Glioma-Post",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zips", nargs="+", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    script_path = Path(__file__).resolve()
    manifest = args.manifest or script_path.parent / "manifest.csv"
    if not manifest.exists():
        manifest = script_path.parent / "Kaggle_35_Training_Notebooks/manifest.csv"
    data_root = args.data_root or default_data_root(script_path)
    rows = read_manifest(manifest)
    truth_cases = index_truth_masks(data_root)
    for zip_path in args.zips:
        assert zip_path.exists(), zip_path
        row = manifest_row_for_zip(zip_path, rows)
        enrich_one(zip_path, row, truth_cases, args.output_dir)


if __name__ == "__main__":
    main()

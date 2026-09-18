#!/usr/bin/env python3
"""Release-blocking static and data-contract audit for the 35 Kaggle notebooks."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "Kaggle_35_Training_Notebooks"
DATA_ROOT = ROOT / "MU-Glioma-Post"
MANIFEST = PACKAGE / "manifest.csv"
CONTRACT = "mu_glioma_detailed_zip_v2_nested"
EXPECTED_RUNS = [f"no{number}" for number in range(1, 36)]
EXPECTED_TARGETS = 6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code(notebook: dict, index: int | None = None) -> str:
    cells = notebook["cells"] if index is None else [notebook["cells"][index]]
    return "\n".join(
        "".join(cell["source"])
        for cell in cells
        if cell["cell_type"] == "code"
    )


def literal_assignments(source: str) -> dict[str, object]:
    values: dict[str, object] = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            pass
    return values


def extract_embedded_split(source: str) -> dict:
    match = re.search(
        r"split_record\s*=\s*json\.loads\(r(?:'''|\"\"\")(?P<json>.*?)(?:'''|\"\"\")\)",
        source,
        flags=re.DOTALL,
    )
    assert match, "Embedded split JSON not found"
    return json.loads(match.group("json"))


def compile_nested_programs(source: str, run_id: str) -> tuple[int, int]:
    tree = ast.parse(source, filename=f"{run_id}:nnunet_setup")
    trainers = 0
    preflights = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "write_text"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and "class nnUNetTrainer_MUGlioma_" in node.args[0].value
        ):
            compile(node.args[0].value, f"{run_id}:custom_trainer", "exec")
            trainers += 1
    return trainers, preflights


def compile_preflight(source: str, context: dict[str, object], run_id: str) -> int:
    tree = ast.parse(source, filename=f"{run_id}:train_and_predict")
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "preflight_program" for target in node.targets):
            continue
        rendered = eval(
            compile(ast.Expression(node.value), f"{run_id}:preflight_fstring", "eval"),
            {"__builtins__": {}},
            context,
        )
        compile(rendered, f"{run_id}:rendered_preflight", "exec")
        count += 1
    return count


def index_local_data() -> tuple[dict[str, dict], list[str]]:
    cases: dict[str, dict] = {}
    missing_masks: list[str] = []
    modalities = ("brain_t1n", "brain_t1c", "brain_t2w", "brain_t2f")
    for patient_dir in sorted(DATA_ROOT.glob("PatientID_*")):
        for timepoint_dir in sorted(patient_dir.glob("Timepoint_*")):
            case_id = f"{patient_dir.name}_{timepoint_dir.name}"
            mask = timepoint_dir / f"{case_id}_tumorMask.nii.gz"
            images = [
                timepoint_dir / f"{case_id}_{modality}.nii.gz"
                for modality in modalities
            ]
            if mask.exists() and all(path.exists() for path in images):
                cases[case_id] = {
                    "patient": patient_dir.name,
                    "mask": mask,
                    "images": images,
                }
            elif all(path.exists() for path in images):
                missing_masks.append(case_id)
    return cases, missing_masks


def validate_split(split: dict, cases: dict[str, dict], missing_masks: list[str]) -> None:
    patient_roles = {
        role: set(split[f"{role}_patients"])
        for role in ("fit_train", "tuning", "outer_test")
    }
    case_roles = {
        role: set(split[f"{role}_cases"])
        for role in ("fit_train", "tuning", "outer_test")
    }
    all_patients = {case["patient"] for case in cases.values()}
    assert not (patient_roles["fit_train"] & patient_roles["tuning"])
    assert not (patient_roles["fit_train"] & patient_roles["outer_test"])
    assert not (patient_roles["tuning"] & patient_roles["outer_test"])
    assert set().union(*patient_roles.values()) == all_patients
    assert sum(map(len, patient_roles.values())) == len(all_patients)
    assert not (case_roles["fit_train"] & case_roles["tuning"])
    assert not (case_roles["fit_train"] & case_roles["outer_test"])
    assert not (case_roles["tuning"] & case_roles["outer_test"])
    assert set().union(*case_roles.values()) == set(cases)
    assert sum(map(len, case_roles.values())) == len(cases)
    for role in patient_roles:
        assert {cases[case_id]["patient"] for case_id in case_roles[role]} == patient_roles[role]
    assert split["missing_masks_excluded"] == missing_masks


def main() -> None:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 35
    assert [row["run_id"] for row in rows] == EXPECTED_RUNS
    assert len({row["notebook"] for row in rows}) == 35
    assert len({row["expected_zip"] for row in rows}) == 35
    assert Counter(row["analysis_role"] for row in rows) == {
        "primary_five_fold_cross_validation": 25,
        "training_seed_robustness_fold1": 10,
    }
    assert Counter(row["framework"] for row in rows) == {"keras": 28, "nnunet": 7}
    assert all(row["zip_contract"] == CONTRACT for row in rows)

    cases, missing_masks = index_local_data()
    assert len(cases) == 594
    assert len({case["patient"] for case in cases.values()}) == 203
    assert missing_masks == [
        "PatientID_0187_Timepoint_3",
        "PatientID_0191_Timepoint_1",
    ]

    split_by_fold: dict[int, dict] = {}
    split_hash_by_fold: dict[int, str] = {}
    for fold in range(1, 6):
        path = PACKAGE / "splits" / f"fold{fold}.json"
        split = json.loads(path.read_text(encoding="utf-8"))
        canonical = json.dumps(split, indent=2)
        assert path.read_text(encoding="utf-8") == canonical
        validate_split(split, cases, missing_masks)
        split_by_fold[fold] = split
        split_hash_by_fold[fold] = hashlib.sha256(canonical.encode()).hexdigest()
    outer_case_counts = Counter(
        case_id
        for split in split_by_fold.values()
        for case_id in split["outer_test_cases"]
    )
    outer_patient_counts = Counter(
        patient
        for split in split_by_fold.values()
        for patient in split["outer_test_patients"]
    )
    assert set(outer_case_counts) == set(cases)
    assert set(outer_case_counts.values()) == {1}
    assert set(outer_patient_counts) == {case["patient"] for case in cases.values()}
    assert set(outer_patient_counts.values()) == {1}

    metric_source = (PACKAGE / "mu_glioma_detailed_metrics.py").read_text(encoding="utf-8")
    metric_inline = metric_source.replace("from __future__ import annotations\n\n", "", 1).rstrip() + "\n"
    compile(metric_source, "mu_glioma_detailed_metrics.py", "exec")
    compile(
        (PACKAGE / "validate_and_enrich_returned_zip.py").read_text(encoding="utf-8"),
        "validate_and_enrich_returned_zip.py",
        "exec",
    )
    freeze = json.loads((PACKAGE / "protocol_freeze.json").read_text(encoding="utf-8"))
    assert freeze["protocol_version"] == CONTRACT
    assert freeze["corrected_outer_test_results_inspected_after_freeze"] is False
    assert freeze["manifest_sha256"] == sha256(MANIFEST)
    assert freeze["metric_helper_sha256"] == sha256(
        PACKAGE / "mu_glioma_detailed_metrics.py"
    )
    assert freeze["fold_split_sha256"] == {
        str(fold): split_hash_by_fold[fold] for fold in range(1, 6)
    }

    code_cell_count = 0
    trainer_count = 0
    preflight_count = 0
    architecture_model_cells: dict[str, set[str]] = defaultdict(set)
    shared_keras_cells: dict[tuple[int, int], set[str]] = defaultdict(set)
    notebook_records = {}

    for row in rows:
        run_id = row["run_id"]
        notebook_path = PACKAGE / row["notebook"]
        assert notebook_path.exists()
        assert sha256(notebook_path) == row["notebook_sha256"]
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert isinstance(notebook.get("metadata"), dict)
        for cell_index, cell in enumerate(notebook["cells"]):
            assert cell["cell_type"] in {"code", "markdown"}
            assert isinstance(cell["source"], list)
            if cell["cell_type"] == "code":
                code_cell_count += 1
                assert cell.get("execution_count") is None
                assert cell.get("outputs", []) == []
                compile("".join(cell["source"]), f"{run_id}:cell{cell_index}", "exec")

        all_code = code(notebook)
        setup_values = literal_assignments(all_code)
        fold = int(row["cv_fold"])
        split = extract_embedded_split(all_code)
        assert split == split_by_fold[fold]
        assert row["split_sha256"] == split_hash_by_fold[fold]
        assert setup_values["RUN_ID"] == run_id
        assert int(setup_values["CV_FOLD"]) == fold
        assert int(setup_values["SPLIT_SEED"]) == int(row["outer_split_seed"])
        assert int(setup_values["INNER_SPLIT_SEED"]) == int(row["inner_split_seed"])
        assert int(setup_values["TRAINING_SEED"]) == int(row["training_seed"])
        assert int(setup_values["EXPECTED_FIT_TRAIN_CASES"]) == int(row["fit_train_cases"])
        assert int(setup_values["EXPECTED_TUNING_CASES"]) == int(row["tuning_cases"])
        assert int(setup_values["EXPECTED_OUTER_TEST_CASES"]) == int(row["outer_test_cases"])
        assert setup_values["EXPECTED_SPLIT_SHA256"] == row["split_sha256"]
        assert CONTRACT in all_code
        assert "outer_test_predictions" in all_code
        assert "outer_test_per_case_metrics.csv" in all_code
        assert "outer_test_per_patient_metrics.csv" in all_code
        assert "outer_test_prediction_manifest.csv" in all_code
        assert "absent_reference_false_positive_summary.csv" in all_code
        assert "artifact_checksums.csv" in all_code

        if row["framework"] == "keras":
            assert len(notebook["cells"]) == 25
            assert setup_values["OUTPUT_NAME"] + "_results.zip" == row["expected_zip"]
            assert 'f"{OUTPUT_NAME}_results.zip"' in all_code
            assert int(setup_values["EPOCHS"]) == 80
            assert float(setup_values["MIN_IMPROVEMENT"]) == 0.002
            assert int(setup_values["EARLY_STOPPING_PATIENCE"]) == 8
            assert int(setup_values["EARLY_STOPPING_START_EPOCH"]) == 20
            assert "train_cases = [case for case in cases if case[\"patient\"] in fit_train_patient_set]" in all_code
            assert "val_cases = tuning_cases" in all_code
            assert "model.fit(\n        train_ds,\n        validation_data=val_ds" in all_code
            assert "class FitCompletionMarker" in all_code
            assert "assert FIT_COMPLETE_PATH.exists()" in all_code
            assert "os.replace(temporary_best_model, BEST_MODEL)" in all_code
            assert "for number, case in enumerate(outer_test_cases, 1):" in all_code
            assert all_code.index("os.replace(temporary_best_model, BEST_MODEL)") < all_code.index(
                "for number, case in enumerate(outer_test_cases, 1):"
            )
            assert metric_inline in all_code
            assert all_code.index('(OUTPUT / "artifact_manifest.json").write_text(') < all_code.index(
                'for path in sorted(OUTPUT.rglob("*")):'
            )
            architecture_model_cells[row["architecture"]].add(code(notebook, 13))
            for cell_index in (5, 9, 11, 17, 19, 21, 23):
                shared_keras_cells[(fold, cell_index)].add(code(notebook, cell_index))
        else:
            assert len(notebook["cells"]) == 22
            assert int(setup_values["MAX_EPOCHS"]) == 150
            assert int(setup_values["MIN_EPOCHS"]) == 100
            assert float(setup_values["MIN_IMPROVEMENT"]) == 0.002
            assert int(setup_values["EARLY_STOPPING_PATIENCE"]) == 15
            assert row["expected_zip"] in all_code
            assert "for number, case in enumerate(modeling_cases, 1):" in all_code
            assert "'train': sorted(fit_train_case_ids)" in all_code
            assert "'val': sorted(tuning_case_ids)" in all_code
            assert "set(modeling_case_ids).isdisjoint(outer_test_case_ids)" in all_code
            assert "'-chk', 'checkpoint_best.pth'" in all_code
            assert "preflight_ok_path = output_root / 'preflight_passed.json'" in all_code
            assert "def outer_predictions_are_complete" in all_code
            assert "processed_case_status" in all_code
            assert metric_inline in all_code
            found_trainers, _ = compile_nested_programs(code(notebook, 2), run_id)
            trainer_count += found_trainers
            context = {
                "DATASET_ID": setup_values["DATASET_ID"],
                "TRAINER_NAME": setup_values["TRAINER_NAME"],
                "model_complexity_path": Path("/tmp/model_complexity.json"),
                "preflight_ok_path": Path("/tmp/preflight_passed.json"),
            }
            preflight_count += compile_preflight(code(notebook, 14), context, run_id)

        notebook_records[run_id] = (row, setup_values, split)

    assert code_cell_count == 378
    assert trainer_count == 7
    assert preflight_count == 7
    assert len(architecture_model_cells) == 4
    assert all(len(cells) == 1 for cells in architecture_model_cells.values())
    assert all(len(cells) == 1 for cells in shared_keras_cells.values())

    architecture_blocks = [(1, 7), (8, 14), (15, 21), (22, 28), (29, 35)]
    for start, end in architecture_blocks:
        block = [notebook_records[f"no{number}"] for number in range(start, end + 1)]
        assert [int(item[0]["cv_fold"]) for item in block] == [1, 2, 3, 4, 5, 1, 1]
        assert [int(item[0]["training_seed"]) for item in block] == [2026] * 5 + [2027, 2028]
        assert block[0][2] == block[5][2] == block[6][2]

    for fold in range(1, 6):
        primary = [
            row for row in rows
            if int(row["cv_fold"]) == fold
            and row["analysis_role"] == "primary_five_fold_cross_validation"
        ]
        assert len(primary) == 5
        assert len({row["split_sha256"] for row in primary}) == 1

    print("PASS: 35 unique manifest rows and expected ZIP names")
    print("PASS: 594 cases / 203 patients match all five locked nested splits")
    print("PASS: every patient and case appears in exactly one primary outer fold")
    print(f"PASS: all {code_cell_count} executable notebook cells compile")
    print("PASS: all 7 embedded nnU-Net trainers and 7 rendered preflight programs compile")
    print("PASS: notebook hashes, run identities, seeds, fold counts, and split hashes match manifest")
    print("PASS: fit/tuning/outer roles are disjoint and exhaustive in every notebook")
    print("PASS: checkpoint, outer-test prediction, detailed metrics, and archive contract hooks present")
    print("PASS: shared metric implementation is byte-identical in all 35 notebooks")


if __name__ == "__main__":
    main()

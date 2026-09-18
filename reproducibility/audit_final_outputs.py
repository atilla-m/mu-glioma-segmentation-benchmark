#!/usr/bin/env python3
"""Fail-closed audit of the completed analysis and manuscript deliverables."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parent
ANALYSIS = PACKAGE / "results" / "final_analysis"
ZIP_ROOT = PACKAGE / "results" / "validated_zips"
MANUSCRIPT = PACKAGE / "manuscript" / "MU_Glioma_Research_Paper.md"
PDF = MANUSCRIPT.with_suffix(".pdf")
DOCX = MANUSCRIPT.with_suffix(".docx")
PRIMARY_TARGET = "Tumor-related region (1-3)"
MODEL_ORDER = [
    "Standard 3D U-Net",
    "Residual 3D U-Net",
    "Controlled 3D V-Net",
    "Controlled 3D U-Net++",
    "nnU-Net v2 3D fullres",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_member(path: Path, suffix: str) -> bytes:
    with zipfile.ZipFile(path) as archive:
        matches = [name for name in archive.namelist() if name.endswith(suffix)]
        assert len(matches) == 1, (path, suffix, matches)
        return archive.read(matches[0])


def main():
    checks = []

    def check(name, condition, detail=""):
        passed = bool(condition)
        checks.append({"check": name, "passed": passed, "detail": str(detail)})
        if not passed:
            raise AssertionError(f"{name}: {detail}")

    zips = sorted(ZIP_ROOT.glob("no*/*.zip"))
    check("35 validated ZIPs are present", len(zips) == 35, len(zips))
    run_ids = {path.parent.name for path in zips}
    check("ZIP run IDs are exactly no1-no35", run_ids == {f"no{i}" for i in range(1, 36)})

    analysis_manifest = json.loads((ANALYSIS / "analysis_manifest.json").read_text())
    for archive in analysis_manifest["validated_archives"]:
        path = ROOT / archive["zip_path"]
        check(
            f"input hash {archive['run_id']}",
            sha256(path) == archive["zip_sha256"],
            archive["zip_sha256"],
        )

    primary = pd.read_csv(ANALYSIS / "primary_patient_metrics.csv")
    check("primary row count", len(primary) == 203 * 5, len(primary))
    for model in MODEL_ORDER:
        group = primary[primary.architecture == model]
        check(f"203 primary patients: {model}", group.patient_id.nunique() == 203)
        check(f"no missing primary Dice: {model}", group.primary_dice.notna().all())

    case_frames = []
    tracker = pd.read_csv(PACKAGE / "results" / "progress" / "MU_Glioma_35_Run_Progress.csv")
    primary_ids = tracker.loc[
        tracker.analysis_role == "primary_five_fold_cross_validation", "run_id"
    ]
    for run_id in primary_ids:
        path = next((ZIP_ROOT / run_id).glob("*.zip"))
        frame = pd.read_csv(io.BytesIO(read_member(path, "outer_test_per_case_metrics.csv")))
        architecture = tracker.loc[tracker.run_id == run_id, "architecture"].iloc[0]
        frame["architecture"] = architecture
        case_frames.append(frame)
    cases = pd.concat(case_frames, ignore_index=True)
    tumor = cases[(cases.target == PRIMARY_TARGET) & (cases.gt_present.astype(str).str.lower() == "true")]
    recomputed = (
        tumor.groupby(["architecture", "patient_id"], as_index=False).dice.mean()
        .rename(columns={"dice": "recomputed"})
    )
    merged = primary.merge(recomputed, on=["architecture", "patient_id"], validate="one_to_one")
    maximum_difference = float(np.max(np.abs(merged.primary_dice - merged.recomputed)))
    check(
        "primary patient Dice reproduces from per-case tables",
        maximum_difference < 1e-12,
        maximum_difference,
    )

    summary = pd.read_csv(ANALYSIS / "primary_model_summary.csv")
    means = primary.groupby("architecture").primary_dice.mean()
    maximum_summary_difference = max(
        abs(row.mean_dice - means[row.architecture]) for row in summary.itertuples()
    )
    check(
        "primary summary means reproduce",
        maximum_summary_difference < 1e-12,
        maximum_summary_difference,
    )

    holdout = json.loads((PACKAGE / "protocol" / "development_holdout_patients.json").read_text())
    check("development holdout has 41 unique patients", len(set(holdout["patients"])) == 41)
    check(
        "complementary sensitivity has 162 patients",
        primary.loc[~primary.patient_id.isin(holdout["patients"]), "patient_id"].nunique() == 162,
    )

    selection = pd.read_csv(ANALYSIS / "qualitative_case_selection.csv")
    check("three qualitative cases selected", len(selection) == 3, len(selection))
    check("qualitative patients are distinct", selection.patient_id.nunique() == 3)
    for row in selection.itertuples(index=False):
        for model in MODEL_ORDER:
            run_id = tracker.loc[
                (tracker.architecture == model)
                & (tracker.cv_fold == int(row.fold))
                & (tracker.analysis_role == "primary_five_fold_cross_validation"),
                "run_id",
            ].iloc[0]
            path = next((ZIP_ROOT / run_id).glob("*.zip"))
            with zipfile.ZipFile(path) as archive:
                count = sum(
                    name.endswith(f"outer_test_predictions/{row.case_id}.nii.gz")
                    for name in archive.namelist()
                )
            check(f"qualitative prediction {row.case_id} {model}", count == 1, count)

    manuscript_text = MANUSCRIPT.read_text(encoding="utf-8")
    check("manuscript has no pending analysis marker", "[PENDING" not in manuscript_text)
    for token in ["0.879", "47.29", "4.76 × 10^-33", "4,160", "Figure 6"]:
        check(f"manuscript contains {token}", token in manuscript_text)

    figures = [
        "study_design.png",
        "primary_performance.png",
        "target_metrics.png",
        "absent_reference_false_positives.png",
        "fold_seed_sensitivity.png",
        "qualitative_predictions.png",
    ]
    for filename in figures:
        path = PACKAGE / "manuscript" / "figures" / filename
        check(f"figure exists and is nonempty: {filename}", path.stat().st_size > 50_000, path.stat().st_size)

    check("PDF exists", PDF.stat().st_size > 500_000, PDF.stat().st_size)
    info = subprocess.run(["pdfinfo", str(PDF)], check=True, capture_output=True, text=True).stdout
    pages = int(next(line.split(":", 1)[1] for line in info.splitlines() if line.startswith("Pages:")).strip())
    check("PDF has at least 10 pages", pages >= 10, pages)
    text_path = Path("/tmp/mu_glioma_final_audit_text.txt")
    subprocess.run(["pdftotext", str(PDF), str(text_path)], check=True)
    pdf_text = text_path.read_text(encoding="utf-8")
    for token in ["Primary out-of-fold", "0.879", "Declaration of generative AI", "References"]:
        check(f"PDF contains {token}", token in pdf_text)

    with zipfile.ZipFile(DOCX) as archive:
        embedded_images = [name for name in archive.namelist() if name.startswith("word/media/")]
    check("DOCX embeds all six figures", len(embedded_images) == 6, embedded_images)

    report = {
        "audit_version": "final_manuscript_audit_v1",
        "date": datetime.now().astimezone().date().isoformat(),
        "status": "PASS",
        "checks_passed": len(checks),
        "checks": checks,
    }
    destination = ANALYSIS / "final_audit_report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(checks)} checks; report written to {destination}")


if __name__ == "__main__":
    main()

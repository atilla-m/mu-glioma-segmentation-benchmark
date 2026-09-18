# Patient-grouped benchmarking of postoperative glioma segmentation pipelines

This repository contains the manuscript and reproducibility materials for
**Nested Patient-Grouped Benchmarking of 3D Deep-Learning Pipelines for
Multiclass Postoperative Glioma Segmentation on Longitudinal MRI**.

The study compares 3D U-Net, Residual 3D U-Net, 3D V-Net, and 3D U-Net++ under
a controlled Keras pipeline and evaluates nnU-Net v2 3D full resolution as a
practical whole-pipeline comparator. The analysis uses 594 eligible longitudinal
MRI examinations from 203 patients in the public MU-Glioma-Post collection.
All partitions are patient grouped, with an inner tuning partition for model
selection and a held-out outer fold for evaluation.

This is a technical internal-validation study. It is not a diagnostic-accuracy
study and does not establish clinical utility or external generalizability.

## Repository contents

| Directory | Contents |
|---|---|
| [`manuscript/`](manuscript/) | Manuscript in DOCX and PDF, plus all manuscript figures |
| [`notebooks/`](notebooks/) | Exact execution notebooks and their SHA-256 manifest |
| [`protocol/`](protocol/) | Prespecified analysis, patient-grouped splits, run matrix, and artifact contract |
| [`results/final_analysis/`](results/final_analysis/) | Patient-level results, statistical comparisons, secondary metrics, and audit records |
| [`results/progress/`](results/progress/) | Complete run tracker in CSV and Excel formats |
| [`results/validation_records/`](results/validation_records/) | Machine-readable validation record for each accepted run |
| [`reproducibility/`](reproducibility/) | Analysis, metric, validation, figure-generation, and audit code |
| [`artifacts/`](artifacts/) | SHA-256 identities for the retained full result archives |
| [`environment/`](environment/) | Analysis dependencies |

## Study design

The primary matrix consists of five model families evaluated across five frozen
outer folds using training seed 2026. Additional fold-1 runs with seeds 2027
and 2028 assess optimization sensitivity without increasing the independent
patient sample size. Checkpoint selection, learning-rate scheduling, and early
stopping use only the inner tuning partition. The outer fold is used only for
full-volume evaluation of the selected checkpoint.

The complete design and fixed run definitions are available in
[`protocol/PRESPECIFIED_ANALYSIS.md`](protocol/PRESPECIFIED_ANALYSIS.md) and
[`protocol/run_matrix_manifest.csv`](protocol/run_matrix_manifest.csv).

## Primary result

The prespecified primary target is the patient-level Dice coefficient for the
combined tumor-related region (labels 1–3). Each estimate contains one
out-of-fold result per patient and architecture.

| Model | Patients | Mean Dice | 95% bootstrap CI |
|---|---:|---:|---:|
| 3D U-Net | 203 | 0.774 | 0.748–0.799 |
| Residual 3D U-Net | 203 | 0.785 | 0.759–0.808 |
| 3D V-Net | 203 | 0.781 | 0.755–0.805 |
| 3D U-Net++ | 203 | 0.761 | 0.735–0.786 |
| nnU-Net v2 3D fullres | 203 | 0.879 | 0.856–0.900 |

nnU-Net changes multiple pipeline components and is therefore interpreted as a
whole-pipeline comparator, not as an architecture-only comparison. Full overlap,
surface, lesion-level, absent-reference, fold, seed, and efficiency results are
reported in the manuscript and under [`results/final_analysis/`](results/final_analysis/).

## Manuscript

- [PDF](manuscript/MU_Glioma_Research_Paper.pdf)
- [DOCX](manuscript/MU_Glioma_Research_Paper.docx)

The manuscript has not yet completed journal peer review. Numerical claims
should be interpreted within the limitations described in the paper.

## Dataset

The original MRI images and reference segmentations are not redistributed here.
They are available from The Cancer Imaging Archive:

- Collection: [MU-Glioma-Post](https://www.cancerimagingarchive.net/collection/mu-glioma-post/)
- Dataset DOI: [10.7937/7K9K-3C83](https://doi.org/10.7937/7K9K-3C83)

The frozen patient-level split definitions used in this study are included in
[`protocol/splits/`](protocol/splits/).

## Reproducibility

The committed CSV, JSON, Excel, and figure files are sufficient to inspect the
reported numerical results without downloading the imaging dataset. To prepare
the analysis environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/requirements-analysis.txt
python reproducibility/audit_execution_notebooks.py
```

The full analysis program reads the validated run archives from
`results/validated_zips/no1` through `results/validated_zips/no35`. Those large
archives contain checkpoints, logs, configurations, detailed metrics, and
full-volume predictions. They are retained by the authors but are not currently
distributed through this repository. Their filenames, sizes, and SHA-256 hashes
are recorded in [`results/validated_result_manifest.csv`](results/validated_result_manifest.csv)
and [`artifacts/validated_archive_checksums.sha256`](artifacts/validated_archive_checksums.sha256).

The exact training notebooks are included for method inspection and replication.
Running them requires a separately obtained copy of MU-Glioma-Post and a
compatible GPU environment. Platform-specific storage paths may need adaptation;
the patient partitions, random seeds, stopping rules, target definitions, and
evaluation rules should remain unchanged for a direct replication.

## Licensing

Original code and notebook code cells are licensed under the MIT License.
Original manuscript text, documentation, tables, and figures are licensed under
CC BY 4.0. The source dataset and third-party software retain their own terms.
See [`LICENSES.md`](LICENSES.md) for the exact scope.

## Citation

Repository citation metadata are provided in [`CITATION.cff`](CITATION.cff).
Any use of the source imaging data must also cite the MU-Glioma-Post dataset and
its associated publication.

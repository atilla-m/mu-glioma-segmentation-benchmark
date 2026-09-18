# MU-Glioma-Post segmentation benchmark

Reproducibility materials for **Nested Patient-Grouped Benchmarking of 3D
Deep-Learning Pipelines for Multiclass Postoperative Glioma Segmentation on
Longitudinal MRI**.

This repository is designed to be usable on its own. It contains the manuscript,
the exact execution notebooks, frozen patient splits, analysis code, compact
results, figures, run records, and checksums. The large validated run archives
are distributed as assets attached to the GitHub release rather than stored in
Git history.

The source MRI dataset is intentionally not redistributed. It remains available
from The Cancer Imaging Archive as MU-Glioma-Post:

- Collection: <https://www.cancerimagingarchive.net/collection/mu-glioma-post/>
- Dataset DOI: <https://doi.org/10.7937/7K9K-3C83>

## Study materials

| Location | Contents |
|---|---|
| `paper/` | Current manuscript draft and publication figures |
| `notebooks/` | Exact notebooks used for the completed runs and their manifest |
| `protocol/` | Frozen analysis plan, run matrix, fold definitions, and artifact contract |
| `results/final_analysis/` | Final patient-level analysis, statistical tables, and audit report |
| `results/progress/` | Human-readable CSV and Excel run tracker |
| `results/validation_records/` | Machine-readable validation record for every run |
| `reproducibility/` | Metric, validation, analysis, figure, and audit programs |
| `release-assets/` | Checksums and instructions for the large result archives |

The primary matrix contains one patient-grouped out-of-fold evaluation for each
model family and outer fold. Additional fold-1 runs with different training
seeds assess optimization sensitivity without increasing the independent sample
size. See `protocol/PRESPECIFIED_ANALYSIS.md` and
`protocol/run_matrix_manifest.csv` for the exact design.

## Read the results without downloading large files

The manuscript, figures, final statistical tables, run tracker, validation
records, and analysis manifest are all committed directly to the repository.
Start with:

1. `paper/MU_Glioma_Research_Paper_Draft.pdf`
2. `results/final_analysis/README.md`
3. `results/final_analysis/primary_model_summary.csv`
4. `results/progress/MU_Glioma_35_Run_Progress.xlsx`

## Reproduce the final analysis

Python 3.11 is recommended. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/requirements-analysis.txt
./release-assets/download_validated_results.sh
python reproducibility/analyze_final_35_runs.py
python reproducibility/audit_final_outputs.py
```

The download script obtains the 35 validated archives from the GitHub release,
checks their SHA-256 hashes, and places them under
`results/validated_zips/no1` through `results/validated_zips/no35`, which is the
layout expected by the analysis programs.

The original MRI data are not required to inspect the committed numerical
results. They are required for training and for rebuilding image-based
qualitative panels. Follow `protocol/DATASET_NOT_INCLUDED.md` and retain the
original TCIA directory layout.

## Run provenance

`notebooks/execution_notebook_manifest.csv` maps each run to the notebook and
platform actually used. `results/validated_result_manifest.csv` records the
architecture, fold, seed, epoch information, score summary, archive size, and
SHA-256 hash for every accepted result.

The notebooks contain no saved execution outputs. Platform-specific absolute
paths in code are part of the original execution configuration and may need to
be changed when rerunning on another system; the frozen patient identities,
folds, seeds, stopping rules, and metric definitions must not be changed if the
goal is an exact replication.

## Status

This is the private author/supervisor review version. Items that must be resolved
before making the repository public are listed in `REVIEW_REQUIRED.md`. No DOI
or external archive is required to use this GitHub package; a DOI archive can be
added later as an optional preservation mirror.

## Citation

Citation metadata are provided in `CITATION.cff`. Until the article receives a
formal citation, cite this repository together with the MU-Glioma-Post source
dataset DOI above.

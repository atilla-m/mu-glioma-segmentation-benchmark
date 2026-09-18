# Final analysis

This directory is the machine-readable basis of the complete manuscript. The
analysis was run only after all 35 archives passed the frozen ZIP contract.

## Primary outputs

- `primary_patient_metrics.csv`: one out-of-fold tumor-union Dice row per
  patient and architecture (203 × 5 rows).
- `primary_model_summary.csv`: equal-patient estimates and 100,000-resample
  bootstrap confidence intervals.
- `primary_friedman_test.csv` and `primary_paired_comparisons.csv`: the frozen
  omnibus and paired inferential analysis.
- `complementary_162_patient_*.csv`: prespecified sensitivity analysis that
  excludes the earlier 41-patient development holdout.
- `paper_numbers.json`: one machine-readable collection of all values used in
  the manuscript.

## Secondary and robustness outputs

- `target_metrics_summary.csv`: uniform class/union overlap, surface, and
  lesion metrics with reference-present denominators and undefined counts.
- `absent_reference_summary.csv`: scan- and patient-level false-positive
  burden, Wilson intervals, predicted volumes, and lesion counts.
- `fold_summary.csv`: descriptive five-fold estimates.
- `seed_fold1_means.csv` and `seed_sensitivity_summary.csv`: fold-1 results for
  seeds 2026, 2027, and 2028 without inflating the patient sample size.
- `run_efficiency.csv` and `efficiency_summary.csv`: run-level and model-level
  resource records; hardware was not balanced, so these are descriptive.
- `qualitative_case_selection.csv`: the deterministic case-selection rule and
  selected cases used for Figure 6.

## Integrity

- `analysis_manifest.json` records the analysis environment, script hash, and
  SHA-256 identity of every input archive.
- `final_audit_report.json` records 88 independent checks, including direct
  reconstruction of primary patient scores from per-case ZIP members, all
  qualitative prediction identities, key manuscript values, PDF readability,
  and embedded DOCX figures.

Reproduce the analysis from the repository root with:

```bash
MPLCONFIGDIR=/tmp/mu-glioma-matplotlib \
  .venv-no10-local/bin/python \
  MU_Glioma_Research_Package/reproducibility/analyze_final_35_runs.py
```

Then run `reproducibility/audit_final_outputs.py` using the same environment.

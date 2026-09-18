# Final analysis outputs

This directory contains the machine-readable numerical basis of the manuscript.
The analysis was finalized only after every accepted run archive passed the
prespecified artifact contract.

## Primary outputs

- `primary_patient_metrics.csv`: one out-of-fold tumor-union Dice row per
  patient and architecture (203 × 5 rows).
- `primary_model_summary.csv`: equal-patient estimates and 100,000-resample
  bootstrap confidence intervals.
- `primary_friedman_test.csv` and `primary_paired_comparisons.csv`: the
  prespecified omnibus and paired inferential analyses.
- `complementary_162_patient_*.csv`: sensitivity analysis excluding the earlier
  41-patient development holdout.
- `paper_numbers.json`: machine-readable collection of values used in the
  manuscript.

## Secondary and robustness outputs

- `target_metrics_summary.csv`: class and union overlap, surface, and lesion
  metrics with reference-present denominators and undefined counts.
- `absent_reference_summary.csv`: scan- and patient-level false-positive burden,
  Wilson intervals, predicted volumes, and lesion counts.
- `fold_summary.csv`: descriptive outer-fold estimates.
- `seed_fold1_means.csv` and `seed_sensitivity_summary.csv`: fold-1 results for
  seeds 2026, 2027, and 2028 without inflating the patient sample size.
- `run_efficiency.csv` and `efficiency_summary.csv`: descriptive resource
  records; hardware was not balanced as an experimental factor.
- `qualitative_case_selection.csv`: deterministic selection record for the
  qualitative figure.

## Integrity

- `analysis_manifest.json` records the analysis environment, script identity,
  and SHA-256 identity of every input archive.
- `final_audit_report.json` records the completed full-artifact audit.
- `../validated_result_manifest.csv` and
  `../../artifacts/validated_archive_checksums.sha256` link the public summaries
  to the retained full archives.

The committed summaries can be inspected directly. A complete rerun of
`reproducibility/analyze_final_35_runs.py` additionally requires the retained
validated archives in `results/validated_zips/no1` through
`results/validated_zips/no35` and the original dataset for rebuilding the
qualitative image panel.

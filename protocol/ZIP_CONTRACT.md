# Returned ZIP contract: `mu_glioma_detailed_zip_v2_nested`

Every one of the 35 results is accepted only as a validated `*_DETAILED.zip`.
Keep the original returned ZIP as immutable source evidence.

## Required source evidence

- exact `patient_split.json` matching the run's SHA-256 in `manifest.csv`;
- all untouched outer-test prediction volumes as pseudonymous `.nii.gz` files;
- best saved model/checkpoint (`.keras` or `.pth`);
- complete training log and learning/convergence plot;
- clean-completion evidence (`fit_complete.json` for Keras or
  `preflight_passed.json` for nnU-Net);
- model/experiment configuration and final training summary;
- training environment versions and computational-efficiency telemetry.

## Required uniform reports

- `outer_test_per_case_metrics.csv`: one row per outer-test scan and target;
- `outer_test_per_patient_metrics.csv`: patient-pooled counts and metrics;
- `final_summary.csv` and `final_summary.json`;
- `absent_reference_false_positive_summary.csv`, including the denominator and
  Wilson 95% confidence interval;
- `outer_test_prediction_manifest.csv`, with file sizes and SHA-256 values;
- `efficiency.json`, including parameter count, training time, inference time,
  checkpoint size, hardware, and peak GPU-memory measurement;
- `metric_definitions.json`, `enrichment_provenance.json`, and
  `source_artifact_inventory.json`;
- `artifact_manifest.json` and `artifact_checksums.csv` covering every other
  file in the ZIP.

## Uniform evaluation definitions

The six targets are labels 1, 2, 3, and 4 individually, tumor-related region
(labels 1-3), and all postoperative regions (labels 1-4). Each receives Dice,
IoU, precision, recall, TP/FP/FN and present/absent flags; HD95 in millimetres;
symmetric average surface distance in millimetres plus both directed values;
surface Dice at 1 mm; and lesion counts and lesion-level sensitivity.

All overlap and surface summaries use the fixed reference-present subset for
that target, shared across models. Reference-absent scans never enter an
overlap mean; they are retained exclusively in the false-positive analysis.

Surface distances use physical NIfTI spacing and `surface-distance==0.1`. When
exactly one mask is empty, HD95 and average surface distances are undefined
(`NaN`) and surface Dice is 0. When both masks are empty, all surface metrics
are `NaN` and excluded from means.

Lesions are 26-connected 3D components. A reference lesion is detected when at
least one predicted voxel overlaps it; a predicted component with no reference
overlap is false-positive. For absent-reference scans, the report retains scan
and patient denominators, predicted voxel/lesion burden, false-positive scan
rate, and its 95% Wilson interval. Training-seed repeats are not treated as new
independent patients when the final cross-run analysis is performed.

## Acceptance command

```bash
python validate_and_enrich_returned_zip.py RETURNED_FILE.zip
```

Mark `detailed_zip_validated=yes` in `manifest.csv` only after the command
finishes successfully and prints `Validated detailed archive`.

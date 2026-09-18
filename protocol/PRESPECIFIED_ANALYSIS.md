# Prespecified analysis plan — freeze before corrected training

Protocol version: `mu_glioma_detailed_zip_v2_nested`. No comparative outer-test
results may be inspected before this plan, the five split files, and all 35
notebook hashes are frozen.

## Run families

- **Primary matrix:** the 25 runs with training seed 2026
  (five architectures by five outer folds).
- **Seed robustness:** the ten fold-1 runs with seeds
  2027 and 2028. They are not pooled
  into the primary estimate and do not increase the independent patient count.
- The four older outer-selected Kaggle runs are supplementary pilots and are
  excluded from both families.

## Primary outcome and statistical unit

The independent unit is the patient. The primary outcome is each patient's
mean timepoint Dice for the tumor-related region (labels 1-3), calculated only
on that patient's reference-present outer-test timepoints. Reference-absent
timepoints use the separate false-positive analysis and never enter an overlap
mean. Each patient contributes at most one out-of-fold primary score per
architecture. Missing/failed predictions are not imputed; their number and
identity are reported.

## Controlled architecture comparison

The four Keras architectures are first compared with a two-sided Friedman test
at alpha 0.05. Only if the omnibus test is significant, the three planned
paired comparisons (Residual U-Net, V-Net, and U-Net++ versus Standard U-Net)
use two-sided Wilcoxon signed-rank tests with Pratt handling of zero differences
and Holm correction across the three tests. Report equal-patient mean and
median scores, median paired differences, rank-biserial effect sizes, adjusted
p-values, and 100,000-resample paired patient-bootstrap percentile 95%
confidence intervals using seed 2026.

## nnU-Net pipeline comparison

The separately framed, prospectively fixed pipeline comparison is nnU-Net
versus Residual U-Net; it is not an architectural attribution. It receives the
same paired descriptive effect and bootstrap interval and a separately labeled
two-sided paired Wilcoxon result. It is not substituted post hoc with whichever
controlled model scores highest.

## Secondary and sensitivity analyses

- Apply the same equal-patient/reference-present rule to class Dice, all-region
  Dice, IoU, precision, recall, HD95, average surface distance, surface Dice at
  1 mm, and lesion sensitivity. Clearly mark multiplicity-unadjusted secondary
  intervals and p-values as exploratory.
- Report absent-reference scan and patient denominators, predicted volume,
  false-positive lesions, false-positive scan rate, and Wilson 95% interval.
- Report fold-level results but do not treat the five folds as the inferential
  sample size.
- Because the earlier 41-patient development holdout was inspected during
  protocol development, repeat the primary descriptive/paired analysis on the
  complementary 162 previously unreported patients as a prespecified
  sensitivity analysis.
- Summarize seed robustness within fold 1 using patient-paired score changes,
  ranges, and standard deviations across the three seeds; do not use seed runs
  as independent observations.
- Report parameter count, training time, inference time, peak GPU memory,
  hardware, failures, and archive/checkpoint size descriptively.

## Interpretation boundary

The study supports an internal technical comparison on MU-Glioma-Post. One
implementation and one fixed hyperparameter configuration per architecture,
the whole-pipeline nature of nnU-Net, lack of external testing, and lack of
prospective clinical evaluation remain explicit limitations.

# Proposed confirmatory research protocol

## Controlled patient-level benchmarking of 3D U-Net variants for multiclass postoperative glioma segmentation

**Status (30 August 2026):** A five-model exploratory comparison and equal-patient reanalysis on one locked patient-grouped development holdout are complete. This document defines the confirmatory cross-validation study that should be approved before further training. The completed holdout results will not be relabeled as independent test or confirmatory evidence.

### Research question and scope

On the MU-Glioma-Post longitudinal MRI collection, do residual or nested 3D U-Net designs improve segmentation—particularly of small postoperative regions—when data, training, and evaluation are controlled? How does the best controlled architecture compare with the self-configuring nnU-Net v2 pipeline?

This is a **five-class 3D semantic-segmentation study**, not a disease-classification or diagnostic study. Each timepoint has four aligned MRI inputs (T1, contrast-enhanced T1, T2, and FLAIR). The target classes are background (0), non-enhancing tumor core (1), surrounding FLAIR hyperintensity/edema (2), enhancing tissue (3), and postoperative resection cavity (4). All classes, including label 4, will be retained.

### Dataset and study population

The audited dataset contains **594 complete labeled timepoints from 203 patients**. Two timepoints without masks are excluded: `PatientID_0187_Timepoint_3` and `PatientID_0191_Timepoint_1`. Every timepoint belonging to one patient must remain in the same partition. The dataset is longitudinal, so the patient—not the scan—is the independent statistical unit.

### Experiments and models

1. **Controlled architecture comparison:** standard 3D U-Net, 3D V-Net, 3D Residual U-Net, and 3D U-Net++. These models will use one shared data, training, inference, and evaluation pipeline.
2. **Practical pipeline benchmark:** the best controlled model will be compared with nnU-Net v2 `3d_fullres`. Because nnU-Net configures preprocessing, architecture, training, inference, and post-processing, this comparison will be described as a **pipeline comparison**, not evidence that one network architecture alone is superior.

### Validation design

A fixed **five-fold patient-grouped cross-validation** is frozen and reused unchanged for every model. Each patient appears in exactly one outer internal-test fold. Inside each outer training fold, 20% of the remaining patients form a deterministic, patient-grouped inner-tuning subset used for learning-rate scheduling, early stopping, and checkpoint selection; the outer fold is excluded from fitting and tuning and is used only after the checkpoint is frozen. For nnU-Net, outer-test images are additionally excluded from dataset fingerprinting and planning. Folds are balanced for patient count, timepoint count, label-1 presence, and absent enhancing/cavity targets. Split files, seeds, and notebook hashes are archived before corrected training. The four earlier adaptive jobs that selected checkpoints on their reporting fold are supplementary pilots and must be rerun with the corrected split roles to enter the primary matrix.

### Fixed controls

Across the four controlled architectures, the following will remain identical: inclusion/exclusion rules; label definitions; input channels and order; orientation, resampling, cropping, and training-set-derived normalization; foreground patch sampling; augmentation; loss function; optimizer and learning-rate schedule; number of training updates; checkpoint-selection rule; random seeds; sliding-window inference; test-time augmentation and post-processing; metric code; software versions; and hardware class. Patch size will be fixed when feasible. If memory differs between models, a predeclared rule may reduce only batch size while preserving the effective batch size through gradient accumulation. Parameter count, peak GPU memory, training time, and inference time will be reported because architectural complexity cannot be made identical.

### Outcomes and metric rules

The **primary outcome** is full-volume Dice for the combined tumor-related region (labels 1–3). Secondary outcomes are Dice for labels 1, 2, 3, and 4 separately; Dice for all postoperative regions (labels 1–4); precision; recall; 95th-percentile Hausdorff distance or surface Dice; absent-target false-positive rate; and computational cost. Patch-level training metrics will not be used as final results.

Metrics will first be calculated per timepoint, then averaged within each patient so patients with more scans do not receive greater statistical weight. For each class, segmentation overlap will be calculated on a fixed reference-present subset shared by all models. Reference-absent scans will be analyzed separately using false-positive frequency and predicted false-positive volume. This avoids model-dependent Dice denominators.

### Statistical analysis and reporting

For each model, every eligible reference-present patient will contribute one out-of-fold primary score. The controlled models will first be compared with a two-sided Friedman test at \(\alpha=0.05\). If significant, the three planned comparisons of Residual U-Net, V-Net, and U-Net++ versus Standard U-Net will use two-sided paired Wilcoxon signed-rank tests with Pratt zero handling and Holm correction. Report patient-level median paired differences, 100,000-resample paired bootstrap 95% confidence intervals (seed 2026), rank-biserial effect sizes, and adjusted p-values. Residual U-Net versus nnU-Net is fixed in advance as the separate whole-pipeline comparison rather than selecting the highest-scoring controlled model after inspection. Fold-level and pooled out-of-fold estimates, evaluated-case counts, failures, efficiency measures, and confidence intervals will be reported. A prespecified sensitivity analysis excludes the 41 previously inspected development-holdout patients. Results support internal comparison on this collection only; no claim of external or clinical validity will be made without independent external testing.

### Exploratory holdout evidence already completed (not confirmatory)

All five completed models used the same patient-separated development holdout: **470 timepoints/162 patients for training and 124 timepoints/41 patients for validation, with zero patient overlap**. Standard U-Net, Residual U-Net, V-Net, and U-Net++ used the shared Keras protocol; nnU-Net remained a separate self-configuring pipeline benchmark. Because the holdout was used for checkpoint selection and reporting, it is not an independent test set.

| Model | Patient-weighted mean Dice | Bootstrap 95% CI | Paired difference |
|---|---:|---:|---:|
| Keras 3D U-Net | 0.7274 | 0.6586–0.7874 | Reference |
| Keras Residual 3D U-Net | 0.7473 | 0.6804–0.8050 | +0.0199 vs standard |
| Controlled 3D V-Net | 0.7391 | 0.6701–0.7992 | +0.0117 vs standard |
| Controlled 3D U-Net++ | 0.7181 | 0.6485–0.7793 | −0.0093 vs standard |
| nnU-Net v2 3d_fullres | 0.8665 | 0.8013–0.9183 | +0.1192 vs residual* |

*Separate whole-pipeline comparison.

The holdout comparison shows feasibility and suggests small patient-level improvements for Residual U-Net and V-Net over standard U-Net, no clear U-Net++ improvement, and substantially stronger performance for nnU-Net as a separate practical pipeline benchmark. Paired patient-bootstrap intervals and exploratory Holm-adjusted comparisons have now been completed from the preserved counts. The analysis remains non-confirmatory because one development holdout was used for checkpoint selection and reporting and no external dataset was evaluated. The nnU-Net report's “best epoch 51” is an indexing artifact; it means the best checkpoint from the completed 50-epoch run was used.

### Approval requested before confirmatory cross-validation

- Approve the two-experiment framing: controlled architectures first, nnU-Net as a separate pipeline benchmark.
- Approve tumor-related Dice (labels 1–3) as the primary outcome.
- Approve five patient-grouped outer folds and patient-level statistical analysis.
- Approve the fixed-control list, secondary metrics, and statistical tests.
- After approval, freeze the folds and rerun all four controlled architectures in every fold before inspecting comparative results.

**Primary references:** [MU-Glioma-Post collection](https://www.cancerimagingarchive.net/collection/mu-glioma-post/); [nnU-Net method](https://www.nature.com/articles/s41592-020-01008-z); [CLAIM 2024 reporting guidance](https://pubs.rsna.org/doi/10.1148/ryai.240300).

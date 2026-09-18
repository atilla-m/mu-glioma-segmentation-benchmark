# Retained full-result artifacts

The complete validated run archives are retained by the authors but are not
currently distributed through this public repository. They contain selected
checkpoints, full-volume predictions, detailed per-case and per-patient metrics,
training logs, configurations, environment records, and checksums.

The public repository preserves two independent identity records:

- `validated_archive_checksums.sha256` lists each archive filename and SHA-256
  digest.
- `../results/validated_result_manifest.csv` records the corresponding run,
  model, fold, seed, platform, size, and validation record.

The committed summary outputs and manuscript figures were produced from these
validated archives. The original MRI dataset is not included.

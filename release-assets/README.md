# Validated result archives

The full validated outputs are too large for ordinary Git history. They are
intended to be attached individually to the GitHub release tagged
`results-v1.0.0`.

There are 35 archives, one per run. Each is below GitHub's per-release-asset size
limit. Collectively they contain the preserved checkpoints, configurations,
logs, full-volume predictions, and detailed case/patient metrics used by the
final analysis.

`original_release_assets.sha256` records the frozen SHA-256 hash of every archive.
`../results/validated_result_manifest.csv` provides the run-level metadata,
including architecture, fold, seed, platform, archive size, and the same hash.

After the assets have been attached to the release, run:

```bash
./release-assets/download_validated_results.sh
```

The script downloads, verifies, and arranges them in the directory structure
expected by the analysis code.

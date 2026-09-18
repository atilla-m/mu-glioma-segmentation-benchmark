# Pre-publication security audit

Audit date: 2026-09-18

## Scope

The selected notebooks, source programs, JSON/CSV/Markdown files, manuscript,
and the text-like members of all 35 proposed release archives were checked
before Git initialization.

## Findings

- All 36 execution notebook files contain zero saved cell outputs.
- No GitHub, OpenAI, AWS, Kaggle, RunPod, Lightning, bearer-token, API-key, or
  password signature was detected in the selected repository files.
- No such credential signature was detected in text-like members of the 35
  validated result archives.
- Eight Kaggle notebooks contain the Kaggle dataset owner/slug used for those
  executions. This is a data-source identifier, not an authentication secret.
- Eleven locally produced result archives contain the first author's local
  absolute project path in `environment.json`. Those archives are not committed
  to Git history; publication remains subject to the supervisor-review decision
  recorded in `REVIEW_REQUIRED.md`.
- Generic cloud paths such as `/workspace/` occur in cloud execution provenance.
  They contain no account secret.

This pattern-based audit reduces accidental disclosure risk but does not replace
manual author review. Any credential that may previously have been exposed
elsewhere should be revoked or rotated independently of this repository.

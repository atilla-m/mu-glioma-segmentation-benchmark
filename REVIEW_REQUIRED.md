# Author and supervisor review checklist

This repository has been prepared for private review. Do not make it public
until every item below has been resolved.

## Manuscript

- Add the corresponding author's postal address and email.
- Have the supervisor or relevant institution confirm the exact ethics statement.
  Public, de-identified data do not automatically prove that a particular
  institution requires no determination or exemption.
- Replace the repository placeholder in the code/model/artifact availability
  statement after the private GitHub repository has its final URL.
- Confirm the target journal's required wording for disclosure of generative AI
  and AI-assisted technologies.

## Repository and artifacts

- Confirm the agreed author order and spelling in `CITATION.cff`.
- Approve the proposed licensing split in `LICENSES.md` before publication.
- Confirm that publicly redistributing trained checkpoints and generated
  prediction volumes complies with the MU-Glioma-Post/TCIA terms.
- Decide whether to preserve or sanitize the local absolute project path found
  in `environment.json` inside 11 locally produced result archives. It is not a
  password or token, and the path is useful provenance, but it contains the
  operating-system username of the first author.
- Confirm that the Kaggle dataset slug in eight execution notebooks may remain.
  It identifies the dataset source used by those notebook runs and contains no
  authentication credential.
- Inspect the private repository on GitHub before changing its visibility.

## Release assets

- Upload the 35 archives represented by
  `release-assets/original_release_assets.sha256` to a release tagged
  `results-v1.0.0` only after the redistribution and path decisions above.
- Run the download and checksum procedure from a clean clone before publication.

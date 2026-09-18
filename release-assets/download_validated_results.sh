#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
release_tag="${1:-results-v1.0.0}"
download_dir="${2:-${repo_root}/.release-downloads/${release_tag}}"
checksum_file="${repo_root}/release-assets/original_release_assets.sha256"
destination_root="${repo_root}/results/validated_zips"

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI (gh) is required: https://cli.github.com/" >&2
  exit 1
fi

mkdir -p "${download_dir}"
gh release download "${release_tag}" \
  --pattern '*.zip' \
  --dir "${download_dir}" \
  --clobber

(
  cd "${download_dir}"
  sha256sum --check "${checksum_file}"
)

mkdir -p "${destination_root}"
for archive in "${download_dir}"/*.zip; do
  filename="$(basename "${archive}")"
  run_id="$(printf '%s\n' "${filename}" | sed -nE 's/^MU_Glioma_(no[0-9]+)_.*/\1/p')"
  if [[ -z "${run_id}" ]]; then
    echo "Cannot infer run ID from ${filename}" >&2
    exit 1
  fi
  mkdir -p "${destination_root}/${run_id}"
  cp -f "${archive}" "${destination_root}/${run_id}/${filename}"
done

count="$(find "${destination_root}" -type f -name '*.zip' | wc -l)"
if [[ "${count}" -ne 35 ]]; then
  echo "Expected 35 validated archives, found ${count}" >&2
  exit 1
fi

echo "Validated result archives are ready under ${destination_root}."

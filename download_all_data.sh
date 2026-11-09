#!/usr/bin/env bash
# Download the complete CUHK-X Small Model Track dataset (train and test).
# First accept the gated dataset conditions and log in with `hf auth login`.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${1:-${ROOT}/data}"
REPO='Kevin-Pal/CUHK-X_Small_Model_Track'
REVISION='ca37a2e9c6a06271309be03bbfbe16f29e40d8b4'

if ! command -v hf >/dev/null 2>&1; then
  printf 'Install the Hugging Face CLI: python3 -m pip install -U huggingface_hub\n' >&2
  exit 2
fi
mkdir -p "$DATA_ROOT"
DATA_ROOT="$(cd -- "$DATA_ROOT" && pwd)"
printf 'Downloading gated dataset %s at %s\n' "$REPO" "$REVISION"
printf 'Destination: %s/Small-Model-Track\n' "$DATA_ROOT"
hf download "$REPO" --repo-type dataset --revision "$REVISION" \
  --local-dir "$DATA_ROOT" --include 'Small-Model-Track/*'

TRACK="$DATA_ROOT/Small-Model-Track"
files=(
  'class_mapping.csv'
  'Training/data/HAR.zip'
  'Testing/data/small_model_track_test.zip'
  'Testing/test_file/test.csv'
  'Testing/test_file/sample_submission.csv'
)
for number in 01 02 03 04 05 06 07 08; do
  files+=("Training/data/HAR.z${number}")
done
for relative in "${files[@]}"; do
  if [[ ! -s "$TRACK/$relative" ]]; then
    printf 'Missing or empty dataset file: %s\n' "$TRACK/$relative" >&2
    exit 1
  fi
done
printf 'All 13 Small-Model-Track data files are present.\n'
printf 'Archives stay compressed to save disk space. See README.md for training-data extraction.\n'
printf 'Inference requires the separate checkpoint identified in final_config.json.\n'

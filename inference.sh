#!/usr/bin/env bash
set -euo pipefail
CUHKX_PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${CUHKX_PACKAGE_ROOT}/code:${CUHKX_PACKAGE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${CUHKX_PYTHON:-python}" -m visual_temporal.release.organizer_entrypoint \
  --config "${CUHKX_PACKAGE_ROOT}/final_config.json" "$@"

#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/env.sh"
if [ "$#" -eq 0 ]; then
  set -- status
fi
TORCH29_PYTHON="${CROWN_PYTHON:-/homes/rliuar/micromamba/envs/torch29/bin/python}"
exec "${TORCH29_PYTHON}" "${REPO_ROOT}/scripts/crown_pipeline.py" "$@"

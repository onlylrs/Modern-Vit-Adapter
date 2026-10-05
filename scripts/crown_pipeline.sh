#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/env.sh"
if [ "$#" -eq 0 ]; then
  set -- status
fi
TORCH29_PYTHON="${CROWN_PYTHON:-${MAMBA_ROOT_PREFIX:-${HOME}/micromamba}/envs/torch29/bin/python}"
if [ -x "${TORCH29_PYTHON}" ]; then
  exec "${TORCH29_PYTHON}" "${REPO_ROOT}/scripts/crown_pipeline.py" "$@"
fi
if [ -z "${CROWN_PYTHON:-}" ] && command -v micromamba >/dev/null 2>&1; then
  exec micromamba run -n torch29 python "${REPO_ROOT}/scripts/crown_pipeline.py" "$@"
fi
echo "torch29 Python not found: ${TORCH29_PYTHON}; set CROWN_PYTHON" >&2
exit 1

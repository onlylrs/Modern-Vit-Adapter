#!/usr/bin/env bash
set -euo pipefail

# Source this file before running training / eval scripts, e.g.:
#   source "/home/rliuar/1_Research/dinov3-adapter-research/env.sh"

if [ -n "${ZSH_VERSION:-}" ]; then
  _REPO_ROOT="$(cd "$(dirname "${(%):-%x}")" && pwd)"
else
  _REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
_OPENMMLAB_ROOT="${_REPO_ROOT}/third_party/openmmlab"

export PYTHONPATH="${_REPO_ROOT}:${_OPENMMLAB_ROOT}/mmcv:${_OPENMMLAB_ROOT}/mmdet:${_OPENMMLAB_ROOT}/mmseg:${PYTHONPATH:-}"

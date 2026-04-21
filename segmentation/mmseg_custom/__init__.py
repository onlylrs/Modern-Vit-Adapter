from pathlib import Path
import sys

import mmcv


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DETECTION_ROOT = _REPO_ROOT / 'detection'
if str(_DETECTION_ROOT) not in sys.path:
    sys.path.insert(0, str(_DETECTION_ROOT))

# MMSegmentation 0.20.x rejects newer MMCV versions at import-time even when
# the APIs this repo uses remain compatible. Keep the shim local to mmseg_custom.
mmcv_version_parts = tuple(int(part) for part in mmcv.__version__.split('.')[:3])
if mmcv_version_parts > (1, 5, 0):
    mmcv.__version__ = '1.5.0'

from .core import *  # noqa: F401,F403
from .datasets import *  # noqa: F401,F403
from .models import *  # noqa: F401,F403

# Copyright (c) Shanghai AI Lab. All rights reserved.
from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .models import *  # noqa: F401,F403

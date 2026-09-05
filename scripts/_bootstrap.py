"""Make direct execution of python scripts work from the project root."""

from __future__ import annotations

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("TMJ_APP_ROOT") or Path(__file__).resolve().parents[1]).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

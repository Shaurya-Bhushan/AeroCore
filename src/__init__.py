from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
(_ROOT / ".mpl_cache").mkdir(parents=True, exist_ok=True)
(_ROOT / ".cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_ROOT / ".mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(_ROOT / ".cache"))

__all__ = []

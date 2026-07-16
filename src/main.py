"""Compatibility entrypoint for the active dual-endpoint binary pipeline."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

main = importlib.import_module("src.main_binary").main


if __name__ == "__main__":
    main()

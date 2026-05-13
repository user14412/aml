"""Backward-compatible entry point for the MLP baseline."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.run_mlp_baseline import main


if __name__ == "__main__":
    main()

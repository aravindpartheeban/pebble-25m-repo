"""CLI wrapper for Stage 2a (Contrastive Pairs) & Stage 2b (Hard-Negative MRL + KL)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pebble_pipeline.cli import stage2_main

if __name__ == "__main__":
    stage2_main()

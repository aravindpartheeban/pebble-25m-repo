"""CLI wrapper for teacher feature extraction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pebble_pipeline.cli import featurize_main

if __name__ == "__main__":
    featurize_main()

"""CLI wrapper for end-to-end config-driven pipeline execution."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pebble_pipeline.cli import pipeline_main

if __name__ == "__main__":
    pipeline_main()

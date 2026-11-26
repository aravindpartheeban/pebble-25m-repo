"""CLI wrapper for standardized static embedding evaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pebble_pipeline.cli import eval_main

if __name__ == "__main__":
    eval_main()

"""CLI wrapper for Stage 0 (Tokenizer) & Stage 1 (Tokenlearn + PCA + SIF)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pebble_pipeline.cli import stage1_main

if __name__ == "__main__":
    stage1_main()

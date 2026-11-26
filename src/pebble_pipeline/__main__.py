"""Module entrypoint: `python -m pebble_pipeline <stage> [args...]`."""

from __future__ import annotations

import sys

from .cli import (
    eval_main,
    featurize_main,
    pipeline_main,
    soup_main,
    stage1_main,
    stage2_main,
)

COMMANDS = {
    "featurize": featurize_main,
    "stage1": stage1_main,
    "stage2": stage2_main,
    "soup": soup_main,
    "eval": eval_main,
    "run": pipeline_main,
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        cmds = ", ".join(COMMANDS)
        print(f"Usage: python -m pebble_pipeline <{cmds}> [options]")
        sys.exit(0 if len(sys.argv) >= 2 else 2)

    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd not in COMMANDS:
        print(f"Unknown subcommand '{cmd}'. Choose from: {', '.join(COMMANDS)}", file=sys.stderr)
        sys.exit(2)
    COMMANDS[cmd](rest)


if __name__ == "__main__":
    main()

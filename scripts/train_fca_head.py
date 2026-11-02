#!/usr/bin/env python3
"""Train a frozen-backbone FCA contract head from cached temporal features."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vpp2.fca.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["train-head", *sys.argv[1:]]))

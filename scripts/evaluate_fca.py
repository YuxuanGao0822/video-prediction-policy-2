#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vpp2.fca.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["evaluate-rollouts", *sys.argv[1:]]))

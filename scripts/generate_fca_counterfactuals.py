#!/usr/bin/env python3
"""Generate same-scene minimal-pair manifests for language-following tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vpp2.fca.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["generate-counterfactual", *sys.argv[1:]]))

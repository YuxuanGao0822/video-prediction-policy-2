#!/usr/bin/env python3
from vpp2.fca.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["evaluate-rollouts", *__import__("sys").argv[1:]]))

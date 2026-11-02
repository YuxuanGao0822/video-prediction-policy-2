"""CLI for contract validation, counterfactual generation and rollout scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .contracts import Contract, load_jsonl
from .counterfactual import generate_cases, load_json, save_jsonl
from .metrics import aggregate, evaluate_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FCA-VPP2 experiment utilities")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-contracts")
    validate.add_argument("input")
    score = sub.add_parser("evaluate-rollouts")
    score.add_argument("--rollouts", required=True)
    score.add_argument("--output", required=True)
    generate = sub.add_parser("generate-counterfactual")
    generate.add_argument("input")
    generate.add_argument("output")
    train = sub.add_parser("train-head")
    train.add_argument("--manifest", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--input-dim", type=int)
    train.add_argument("--hidden-dim", type=int, default=256)
    train.add_argument("--num-phases", type=int, default=16)
    train.add_argument("--num-violations", type=int, default=1)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--valid-fraction", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-contracts":
        rows = load_jsonl(args.input)
        for index, row in enumerate(rows):
            Contract.from_dict(row.get("contract", row))
        print(json.dumps({"valid": True, "num_contracts": len(rows)}))
        return 0
    if args.command == "evaluate-rollouts":
        result = evaluate_file(args.rollouts, args.output)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "train-head":
        from .train import train_head
        result = train_head(
            args.manifest, args.output, input_dim=args.input_dim,
            hidden_dim=args.hidden_dim, num_phases=args.num_phases,
            num_violations=args.num_violations, epochs=args.epochs,
            batch_size=args.batch_size, learning_rate=args.lr,
            valid_fraction=args.valid_fraction, seed=args.seed, device=args.device,
        )
        print(json.dumps(result, indent=2))
        return 0
    payload = load_json(args.input)
    specs = payload if isinstance(payload, list) else payload.get("scenes", payload.get("cases", []))
    cases = generate_cases(specs)
    save_jsonl((case.to_dict() for case in cases), args.output)
    print(json.dumps({"num_cases": len(cases), "output": args.output}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

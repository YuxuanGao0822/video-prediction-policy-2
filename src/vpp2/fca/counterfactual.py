"""Counterfactual/minimal-pair manifest generation for FCA experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass
class CounterfactualCase:
    case_id: str
    scene_id: str
    instruction: str
    variant: str
    changed_atom: str
    contract: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_cases(specs: Iterable[Mapping[str, Any]]) -> list[CounterfactualCase]:
    result = []
    for spec in specs:
        scene = str(spec["scene_id"])
        base = dict(spec.get("base", {}))
        for index, variant in enumerate(spec.get("variants", [])):
            variant = dict(variant)
            contract = {**dict(base.get("contract", {})), **dict(variant.get("contract", {}))}
            result.append(CounterfactualCase(
                case_id=f"{scene}:{index}", scene_id=scene,
                instruction=str(variant.get("instruction", base.get("instruction", ""))),
                variant=str(variant.get("name", index)),
                changed_atom=str(variant.get("changed_atom", "unknown")),
                contract=contract,
                metadata={**dict(base.get("metadata", {})), **dict(variant.get("metadata", {}))},
            ))
    return result


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_jsonl(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

"""Typed contract and event normalization for FCA-VPP2 experiments.

The module deliberately depends only on the Python standard library so that it can
be used to score simulator logs on a server without importing the VPP2 model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(x) for x in value]


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip().lower()
    return value or None


def _clean_list(value: Any) -> tuple[str, ...]:
    return tuple(item for item in (_clean(x) for x in _as_list(value)) if item)

@dataclass(frozen=True)
class Contract:
    """A planner-to-policy subtask contract.

    ``required_order`` contains event names in precedence order.  ``forbidden``
    contains symbolic constraints such as ``touch:blue_cup`` or ``actor:right_arm``.
    """

    subtask_id: str = "0"
    object: str | None = None
    part: str | None = None
    actor: str | None = None
    goal: str | None = None
    required_order: tuple[str, ...] = ()
    manner: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    terminal_predicate: str | None = None
    expected_duration: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | str | "Contract") -> "Contract":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, Mapping):
            raise TypeError(f"contract must be a mapping or JSON object, got {type(value)!r}")
        order = value.get("required_order", value.get("order", value.get("events", [])))
        return cls(
            subtask_id=str(value.get("subtask_id", value.get("id", "0"))),
            object=_clean(value.get("object", value.get("target"))),
            part=_clean(value.get("part")),
            actor=_clean(value.get("actor", value.get("arm"))),
            goal=_clean(value.get("goal")),
            required_order=_clean_list(order),
            manner=_clean_list(value.get("manner", value.get("style"))),
            forbidden=_clean_list(value.get("forbidden", value.get("exclusions"))),
            terminal_predicate=_clean(value.get("terminal_predicate", value.get("terminal"))),
            expected_duration=(
                None if value.get("expected_duration") is None
                else float(value["expected_duration"])
            ),
            metadata=dict(value.get("metadata", {})),
        )

    def validate(self) -> None:
        """Validate the contract before it is handed to a policy/controller."""
        if not self.subtask_id.strip():
            raise ValueError("subtask_id must be non-empty")
        if len(set(self.required_order)) != len(self.required_order):
            raise ValueError("required_order contains duplicate events")
        if self.expected_duration is not None and self.expected_duration <= 0:
            raise ValueError("expected_duration must be positive")
        if self.terminal_predicate is None and self.goal is None:
            raise ValueError("contract needs terminal_predicate or goal")

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable representation for rollout logs."""
        result: dict[str, Any] = {
            "subtask_id": self.subtask_id,
            "object": self.object,
            "part": self.part,
            "actor": self.actor,
            "goal": self.goal,
            "required_order": list(self.required_order),
            "manner": list(self.manner),
            "forbidden": list(self.forbidden),
            "terminal_predicate": self.terminal_predicate,
        }
        if self.expected_duration is not None:
            result["expected_duration"] = self.expected_duration
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result

    def atoms(self) -> dict[str, list[str]]:
        return {
            "object": [self.object] if self.object else [],
            "part": [self.part] if self.part else [],
            "actor": [self.actor] if self.actor else [],
            "goal": [self.goal] if self.goal else [],
            "manner": list(self.manner),
            "forbidden": list(self.forbidden),
            "terminal": [self.terminal_predicate] if self.terminal_predicate else [],
        }

@dataclass(frozen=True)
class Event:
    name: str
    verb: str | None = None
    object: str | None = None
    part: str | None = None
    actor: str | None = None
    t_start: float | None = None
    t_end: float | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | str) -> "Event":
        if isinstance(value, str):
            value = {"name": value}
        if not isinstance(value, Mapping):
            raise TypeError(f"event must be a mapping or string, got {type(value)!r}")
        return cls(
            name=str(value.get("name", value.get("event", value.get("verb", "unknown")))),
            verb=_clean(value.get("verb", value.get("action"))),
            object=_clean(value.get("object", value.get("target"))),
            part=_clean(value.get("part")),
            actor=_clean(value.get("actor", value.get("arm"))),
            t_start=(None if value.get("t_start") is None else float(value["t_start"])),
            t_end=(None if value.get("t_end") is None else float(value["t_end"])),
            attributes=dict(value.get("attributes", value.get("attrs", {}))),
        )

def parse_events(value: Any) -> list[Event]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        value = value.get("events", [])
    return [Event.from_dict(x) for x in value]


@dataclass(frozen=True)
class ContractSequence:
    """A planner output containing ordered subtask contracts."""

    contracts: tuple[Contract, ...]
    task_id: str | None = None
    instruction: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | Sequence[Any] | "ContractSequence") -> "ContractSequence":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            raw = value.get("contracts", value.get("subtasks", value.get("steps", [])))
            task_id = _clean(value.get("task_id", value.get("scene_id")))
            instruction = value.get("instruction")
        else:
            raw, task_id, instruction = value, None, None
        contracts = tuple(Contract.from_dict(item) for item in raw)
        if not contracts:
            raise ValueError("contract sequence must contain at least one contract")
        for contract in contracts:
            contract.validate()
        return cls(contracts, task_id=task_id, instruction=None if instruction is None else str(instruction))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"contracts": [contract.to_dict() for contract in self.contracts]}
        if self.task_id is not None:
            result["task_id"] = self.task_id
        if self.instruction is not None:
            result["instruction"] = self.instruction
        return result

def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: rollout must be a JSON object")
            records.append(value)
    return records

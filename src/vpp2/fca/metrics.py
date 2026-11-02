"""Offline metrics for the FCA-VPP2 rollout log format."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import Contract, Event, load_jsonl, parse_events


def _safe(num: float, den: float) -> float:
    return 0.0 if den == 0 else float(num) / float(den)

def _f1(gold: Iterable[str], pred: Iterable[str]) -> float:
    g, p = set(str(x).lower() for x in gold), set(str(x).lower() for x in pred)
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    overlap = len(g & p)
    precision, recall = _safe(overlap, len(p)), _safe(overlap, len(g))
    return _safe(2 * precision * recall, precision + recall)

def _field_match(expected: str | None, observed: Iterable[str | None]) -> float:
    if expected is None:
        return 1.0
    return 1.0 if expected.lower() in {str(x).lower() for x in observed if x is not None} else 0.0

def planner_intent_score(predicted: Contract, gold: Contract) -> float:
    """Score the planner contract against an oracle contract."""
    fields = [
        _field_match(gold.object, [predicted.object]),
        _field_match(gold.part, [predicted.part]),
        _field_match(gold.actor, [predicted.actor]),
        _field_match(gold.goal, [predicted.goal]),
        _f1(gold.required_order, predicted.required_order),
        _f1(gold.manner, predicted.manner),
        _f1(gold.forbidden, predicted.forbidden),
        _field_match(gold.terminal_predicate, [predicted.terminal_predicate]),
    ]
    return sum(fields) / len(fields)

def atomic_f1(contract: Contract, events: list[Event]) -> float:
    """Score object/part/actor/goal/manner atoms from an observed event trace."""
    if not events:
        return 0.0
    event_objects = [e.object for e in events]
    event_parts = [e.part for e in events]
    event_actors = [e.actor for e in events]
    event_verbs = [e.verb or e.name for e in events]
    event_manner = []
    for e in events:
        event_manner.extend(str(x) for x in e.attributes.get("manner", []))
        if isinstance(e.attributes.get("manner"), str):
            event_manner.append(e.attributes["manner"])
    scores = [
        _field_match(contract.object, event_objects),
        _field_match(contract.part, event_parts),
        _field_match(contract.actor, event_actors),
        _field_match(contract.goal, event_verbs),
        _f1(contract.manner, event_manner),
    ]
    return sum(scores) / len(scores)

def order_score(contract: Contract, events: list[Event]) -> float:
    if len(contract.required_order) <= 1:
        return 1.0
    names = [e.name.lower() for e in events]
    positions = []
    for required in contract.required_order:
        required = required.lower()
        positions.append(next((i for i, name in enumerate(names) if name == required), None))
    valid_edges = total_edges = 0
    for left, right in zip(positions, positions[1:]):
        total_edges += 1
        if left is not None and right is not None and left < right:
            valid_edges += 1
    return _safe(valid_edges, total_edges)

def temporal_iou(contract: Contract, events: list[Event], tolerance: float = 1.0) -> float:
    if contract.expected_duration is None or not events:
        return 1.0
    observed_end = max((e.t_end for e in events if e.t_end is not None), default=None)
    if observed_end is None:
        return 0.0
    expected = max(float(contract.expected_duration), 1e-6)
    error = abs(observed_end - expected)
    return max(0.0, 1.0 - error / max(expected + tolerance, 1e-6))

def violation_rate(record: Mapping[str, Any], contract: Contract, events: list[Event]) -> float:
    explicit = record.get("violations")
    if explicit is not None:
        count = len(explicit) if isinstance(explicit, list) else int(bool(explicit))
        denom = max(len(contract.forbidden), 1)
        return min(1.0, count / denom)
    hits = 0
    for event in events:
        attrs = event.attributes
        hits += int(bool(attrs.get("forbidden_violation", attrs.get("violation", False))))
    return _safe(hits, max(len(events), 1))

def expected_calibration_error(confidences: list[float], labels: list[bool], bins: int = 10) -> float:
    if not confidences:
        return 0.0
    buckets: list[list[int]] = [[] for _ in range(bins)]
    for i, confidence in enumerate(confidences):
        index = min(bins - 1, max(0, int(float(confidence) * bins)))
        buckets[index].append(i)
    error = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        mean_conf = sum(float(confidences[i]) for i in bucket) / len(bucket)
        mean_label = sum(bool(labels[i]) for i in bucket) / len(bucket)
        error += len(bucket) / len(confidences) * abs(mean_conf - mean_label)
    return float(error)

def evaluate_rollout(record: Mapping[str, Any]) -> dict[str, float | str]:
    predicted_raw = record.get("predicted_contract", record.get("contract", {}))
    predicted = Contract.from_dict(predicted_raw)
    gold = Contract.from_dict(record.get("gold_contract", predicted_raw))
    events = parse_events(record.get("observed_events", record.get("events", [])))
    predicted_events = parse_events(record.get("predicted_events", []))
    intent = planner_intent_score(predicted, gold)
    execution = atomic_f1(gold, events)
    consistency = atomic_f1(gold, predicted_events) if predicted_events else 0.0
    false_advance = float(bool(record.get("advanced_before_terminal", False)))
    terminal = float(bool(record.get("terminal_reached", record.get("success", False))))
    recovery = record.get("recovery_success")
    recovery_value = 0.0 if recovery is None else float(bool(recovery))
    violation = violation_rate(record, gold, events)
    order = order_score(gold, events)
    temporal = temporal_iou(gold, events)
    contract_success = float(terminal and violation == 0.0 and order >= 1.0)
    confidence_trace = record.get("confidence_trace", record.get("confidences", []))
    label_trace = record.get("confidence_labels", record.get("labels", []))
    calibration = expected_calibration_error(
        [float(x) for x in confidence_trace],
        [bool(x) for x in label_trace],
    ) if confidence_trace and len(confidence_trace) == len(label_trace) else 0.0
    return {
        "scene_id": str(record.get("scene_id", record.get("episode_id", "unknown"))),
        "planner_intent_score": intent,
        "execution_following_score": execution,
        "prediction_action_consistency": consistency,
        "contract_success": contract_success,
        "terminal_success": terminal,
        "atomic_f1": execution,
        "order_score": order,
        "temporal_iou": temporal,
        "violation_rate": violation,
        "false_advance_rate": false_advance,
        "recovery_success": recovery_value,
        "calibration_ece": calibration,
    }

def aggregate(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    numeric = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if key != "scene_id" and isinstance(value, (int, float)):
                numeric[key].append(float(value))
    return {key: sum(values) / len(values) for key, values in sorted(numeric.items()) if values}

def evaluate_file(input_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    records = load_jsonl(input_path)
    rows = [evaluate_rollout(record) for record in records]
    result = {"num_rollouts": len(rows), "metrics": aggregate(rows), "per_rollout": rows}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output / "per_rollout.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return result

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(evaluate_file(args.rollouts, args.output), indent=2))

if __name__ == "__main__":
    main()

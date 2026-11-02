"""Contract verification and closed-loop handoff for FCA-VPP2.

The monitor consumes model/simulator-neutral state dictionaries, so it can be
used offline on logged rollouts or online around ``RoboDojoVPP2Policy``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Callable, Mapping, Sequence

from .contracts import Contract


@dataclass
class MonitorDecision:
    action: str  # EXECUTE, ADVANCE, REPLAN, RETRY, ABORT, ASK, DONE
    reason: str
    score: float
    terminal: bool = False
    violation: bool = False
    uncertainty: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)


class PredicateRegistry:
    """Small deterministic predicate registry for simulator/robot state logs."""

    _CALL = re.compile(r"^(?P<name>[A-Za-z_][\w]*)\((?P<args>.*)\)$")

    def __call__(self, predicate: str | None, state: Mapping[str, Any] | None) -> bool:
        text = str(predicate or "").strip()
        state = state or {}
        if not text:
            return False
        predicates = state.get("predicates", {})
        if text in predicates:
            return bool(predicates[text])
        if text in state:
            return bool(state[text])
        match = self._CALL.match(text)
        if match:
            name = match.group("name").lower()
            args = [x.strip().strip("'\"") for x in match.group("args").split(",") if x.strip()]
            if name in {"grasped", "released", "wiped", "moved", "visible", "lifted", "near"} and args:
                value = state.get(name, {})
                return bool(value.get(args[0], False) if isinstance(value, Mapping) else value)
            if name in {"inside", "in", "on", "touching", "contact"} and len(args) >= 2:
                relation = state.get(name, {})
                return bool(isinstance(relation, Mapping) and relation.get(args[0]) == args[1])
        if "=" in text:
            key, expected = (x.strip() for x in text.split("=", 1))
            return str(state.get(key, "")).lower() == expected.lower()
        return False


class ContractMonitor:
    def __init__(
        self,
        contract: Contract,
        *,
        terminal_threshold: float = 0.8,
        violation_threshold: float = 0.5,
        uncertainty_threshold: float = 0.45,
        evaluator: Callable[[str | None, Mapping[str, Any] | None], bool] | None = None,
    ) -> None:
        self.contract = contract
        self.terminal_threshold = float(terminal_threshold)
        self.violation_threshold = float(violation_threshold)
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.evaluator = evaluator or PredicateRegistry()
        self.history: list[MonitorDecision] = []

    def state_score(self, state: Mapping[str, Any] | None) -> dict[str, Any]:
        constraints = [self.evaluator(item, state) for item in self.contract.manner]
        terminal = self.evaluator(self.contract.terminal_predicate or self.contract.goal, state)
        forbidden = [self.evaluator(item, state) for item in self.contract.forbidden]
        violations = [item for item, hit in zip(self.contract.forbidden, forbidden) if hit]
        score = (float(terminal) + sum(map(float, constraints))) / max(1, 1 + len(constraints))
        # A rollout can expose a running event trace.  Checking the order here
        # makes the gate useful even when no learned head is available.
        has_order_trace = bool(state is not None and "completed_events" in state)
        order = state.get("completed_events", []) if state else []
        order = [str(item).strip().lower() for item in order]
        order_ok = True
        if has_order_trace and self.contract.required_order:
            positions = [next((i for i, item in enumerate(order) if item == name), None) for name in self.contract.required_order]
            order_ok = all(left is not None and right is not None and left < right for left, right in zip(positions, positions[1:]))
        return {
            "terminal": terminal,
            "constraints": constraints,
            "forbidden": forbidden,
            "violations": violations,
            "order_ok": order_ok,
            "score": score if order_ok else score * 0.5,
        }

    def prediction_score(self, predicted_states: Sequence[Mapping[str, Any]] | None, uncertainty: float | None = None) -> dict[str, Any]:
        if not predicted_states:
            return {"score": 0.0, "terminal": False, "uncertainty": 0.0, "violation": False, "terminal_frame": None, "available": False}
        rows = [self.state_score(state) for state in predicted_states]
        best_index, best = max(enumerate(rows), key=lambda pair: pair[1]["score"])
        return {
            "score": float(best["score"]),
            "terminal": any(row["terminal"] for row in rows),
            "terminal_frame": next((i for i, row in enumerate(rows) if row["terminal"]), None),
            "uncertainty": float(1.0 - best["score"] if uncertainty is None else uncertainty),
            "violation": any(row["violations"] for row in rows),
            "best_frame": best_index,
            "available": True,
        }

    def step(self, observed_state: Mapping[str, Any] | None, *, predicted_states: Sequence[Mapping[str, Any]] | None = None, predicted_uncertainty: float | None = None) -> MonitorDecision:
        observed = self.state_score(observed_state)
        predicted = self.prediction_score(predicted_states, predicted_uncertainty)
        uncertainty = max(
            float(predicted["uncertainty"]),
            0.0 if predicted_uncertainty is None else float(predicted_uncertainty),
        )
        if observed["violations"] or predicted["violation"]:
            decision = MonitorDecision("REPLAN", "forbidden_or_abort_condition", observed["score"], violation=True, uncertainty=uncertainty, diagnostics={"observed": observed, "predicted": predicted})
        elif observed["terminal"] and observed["score"] >= self.terminal_threshold:
            decision = MonitorDecision("ADVANCE", "terminal_predicate_satisfied", observed["score"], terminal=True, uncertainty=uncertainty, diagnostics={"observed": observed, "predicted": predicted})
        elif uncertainty >= self.uncertainty_threshold:
            decision = MonitorDecision("ASK", "uncertain_contract_state", observed["score"], uncertainty=uncertainty, diagnostics={"observed": observed, "predicted": predicted})
        elif predicted.get("available", False) and predicted["score"] < self.violation_threshold:
            decision = MonitorDecision("REPLAN", "predicted_future_misses_contract", predicted["score"], uncertainty=uncertainty, diagnostics={"observed": observed, "predicted": predicted})
        else:
            decision = MonitorDecision("EXECUTE", "contract_consistent", predicted["score"], uncertainty=uncertainty, diagnostics={"observed": observed, "predicted": predicted})
        self.history.append(decision)
        return decision


class FCAController:
    """State machine for a sequence of planner subtasks.

    The caller supplies the predicted future state sequence and the observed
    simulator/robot state after each action chunk.  No VPP2 internals are
    modified, preserving the paper's official evaluation path.
    """

    def __init__(self, contracts: Sequence[Contract], *, max_replans: int | None = None, **monitor_kwargs: Any) -> None:
        if not contracts:
            raise ValueError("FCAController requires at least one contract")
        self.contracts = list(contracts)
        self.monitors = [ContractMonitor(contract, **monitor_kwargs) for contract in self.contracts]
        self.index = 0
        self.replans = 0
        self.advances = 0
        self.max_replans = None if max_replans is None else int(max_replans)

    @property
    def done(self) -> bool:
        return self.index >= len(self.contracts)

    @property
    def current_contract(self) -> Contract:
        if self.done:
            raise RuntimeError("all FCA subtasks are complete")
        return self.contracts[self.index]

    def observe(self, state: Mapping[str, Any] | None, *, predicted_states: Sequence[Mapping[str, Any]] | None = None, predicted_uncertainty: float | None = None) -> MonitorDecision:
        if self.done:
            return MonitorDecision("DONE", "all_subtasks_complete", 1.0, terminal=True)
        decision = self.monitors[self.index].step(state, predicted_states=predicted_states, predicted_uncertainty=predicted_uncertainty)
        if decision.action == "ADVANCE":
            self.index += 1
            self.advances += 1
        elif decision.action in {"REPLAN", "RETRY", "ASK"}:
            self.replans += 1
            if self.max_replans is not None and self.replans > self.max_replans:
                decision = MonitorDecision(
                    "ABORT", "max_replans_exceeded", decision.score,
                    violation=decision.violation, uncertainty=decision.uncertainty,
                    diagnostics=decision.diagnostics,
                )
        return decision

    @staticmethod
    def should_imagine(decision: MonitorDecision, *, mode: str = "selective") -> bool:
        mode = mode.lower()
        if mode == "always":
            return True
        if mode == "never":
            return False
        return decision.action in {"ASK", "REPLAN"} or decision.uncertainty >= 0.45

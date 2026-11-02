"""Runtime adapters for connecting FCA to an existing VPP2 action sampler."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .monitor import FCAController, MonitorDecision


@dataclass(frozen=True)
class ChunkGate:
    """Decision made before sending an action chunk to the robot/simulator."""

    decision: MonitorDecision
    execute: bool
    imagine: bool
    chunk_steps: int


class FCAActionAdapter:
    """Keep VPP2 inference unchanged while adding contract-aware gating.

    ``observe`` is called after each executed chunk.  The adapter never assumes
    the action representation: ``trim_chunk`` below handles tensors, arrays,
    lists, or objects supporting slicing.
    """

    def __init__(self, controller: FCAController, *, execute_chunk_steps: int = 8, min_chunk_steps: int = 1, imagination_mode: str = "selective"):
        if execute_chunk_steps < 1 or min_chunk_steps < 1 or min_chunk_steps > execute_chunk_steps:
            raise ValueError("require 1 <= min_chunk_steps <= execute_chunk_steps")
        self.controller = controller
        self.execute_chunk_steps = int(execute_chunk_steps)
        self.min_chunk_steps = int(min_chunk_steps)
        self.imagination_mode = imagination_mode

    def observe(self, state: Mapping[str, Any] | None, *, predicted_states: Sequence[Mapping[str, Any]] | None = None, predicted_uncertainty: float | None = None) -> ChunkGate:
        decision = self.controller.observe(state, predicted_states=predicted_states, predicted_uncertainty=predicted_uncertainty)
        imagine = self.controller.should_imagine(decision, mode=self.imagination_mode)
        if decision.action in {"REPLAN", "ASK", "ABORT", "DONE"}:
            steps = self.min_chunk_steps if decision.action in {"REPLAN", "ASK"} else 0
        else:
            steps = self.execute_chunk_steps
        return ChunkGate(decision, execute=steps > 0, imagine=imagine, chunk_steps=steps)

    @staticmethod
    def trim_chunk(chunk: Any, steps: int) -> Any:
        if steps <= 0:
            return chunk[:0]
        return chunk[:steps]


def run_logged_rollout(adapter: FCAActionAdapter, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Replay logged observations and return decisions for diagnostics/tests."""
    output = []
    for row in rows:
        gate = adapter.observe(
            row.get("state", {}),
            predicted_states=row.get("predicted_states"),
            predicted_uncertainty=row.get("predicted_uncertainty"),
        )
        output.append({"action": gate.decision.action, "reason": gate.decision.reason, "score": gate.decision.score, "execute": gate.execute, "imagine": gate.imagine, "chunk_steps": gate.chunk_steps})
        if gate.decision.action == "ABORT":
            break
    return output

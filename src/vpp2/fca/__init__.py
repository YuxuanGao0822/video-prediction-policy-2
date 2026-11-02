"""Future-Contract Alignment (FCA) utilities for VPP2."""
from .contracts import Contract, ContractSequence, Event, load_jsonl, parse_events
from .metrics import aggregate, evaluate_file, evaluate_rollout
from .monitor import ContractMonitor, FCAController, MonitorDecision, PredicateRegistry
from .rollout import ChunkGate, FCAActionAdapter

__all__ = [
    "Contract", "ContractSequence", "Event", "load_jsonl", "parse_events", "aggregate", "evaluate_file",
    "evaluate_rollout", "ContractMonitor", "FCAController", "MonitorDecision", "PredicateRegistry",
    "ChunkGate", "FCAActionAdapter",
]

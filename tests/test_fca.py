import tempfile
import unittest
from pathlib import Path

from vpp2.fca.contracts import Contract, ContractSequence, Event
from vpp2.fca.metrics import evaluate_rollout
from vpp2.fca.monitor import ContractMonitor, FCAController
from vpp2.fca.rollout import FCAActionAdapter


class FCATest(unittest.TestCase):
    def test_contract_and_monitor(self):
        contract = Contract.from_dict({
            "subtask_id": "x",
            "object": "cup",
            "goal": "grasped",
            "terminal_predicate": "grasped(cup)",
            "required_order": ["approach", "grasp"],
        })
        monitor = ContractMonitor(contract, uncertainty_threshold=0.95)
        self.assertEqual(monitor.step({"grasped": {"cup": False}}).action, "EXECUTE")
        self.assertEqual(monitor.step({"grasped": {"cup": True}}).action, "ADVANCE")
        controller = FCAController([contract], uncertainty_threshold=0.95)
        self.assertEqual(controller.observe({"grasped": {"cup": True}}).action, "ADVANCE")

    def test_metrics(self):
        row = evaluate_rollout({
            "predicted_contract": {"object": "cup", "goal": "grasp", "required_order": ["approach", "grasp"]},
            "gold_contract": {"object": "cup", "goal": "grasp", "required_order": ["approach", "grasp"]},
            "observed_events": [
                {"name": "approach", "object": "cup"},
                {"name": "grasp", "object": "cup", "verb": "grasp"},
            ],
            "terminal_reached": True,
            "advanced_before_terminal": False,
            "violations": [],
        })
        self.assertEqual(row["false_advance_rate"], 0.0)
        self.assertGreater(row["contract_success"], 0.0)

    def test_sequence_and_selective_gate(self):
        sequence = ContractSequence.from_dict({"contracts": [{"id": "s0", "goal": "done", "terminal": "done"}]})
        controller = FCAController(sequence.contracts, max_replans=1)
        adapter = FCAActionAdapter(controller, execute_chunk_steps=8, min_chunk_steps=2)
        gate = adapter.observe({"done": False}, predicted_uncertainty=0.9)
        self.assertTrue(gate.imagine)
        self.assertEqual(gate.chunk_steps, 2)
        gate = adapter.observe({"done": True})
        self.assertEqual(gate.decision.action, "ADVANCE")
        self.assertTrue(controller.done)

    def test_order_trace_and_ece(self):
        contract = Contract.from_dict({"goal": "place", "terminal": "placed", "required_order": ["grasp", "place"]})
        monitor = ContractMonitor(contract)
        self.assertEqual(monitor.step({"placed": True, "completed_events": ["grasp", "place"]}).action, "ADVANCE")
        row = evaluate_rollout({"contract": contract.to_dict(), "events": [], "confidence_trace": [0.9, 0.1], "confidence_labels": [True, False]})
        self.assertGreaterEqual(row["calibration_ece"], 0.0)


if __name__ == "__main__":
    unittest.main()

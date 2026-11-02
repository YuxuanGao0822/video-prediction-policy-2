"""Dataset utilities for the frozen-backbone FCA contract head.

The format is intentionally model-agnostic: each JSONL row contains cached
``[T,D]`` features (inline or in a ``.pt``/``.npy`` file) and frame labels.
This lets experiments run on a server without importing the 14B VPP2 model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.utils.data import Dataset


def _load_tensor(value: Any, root: Path) -> Tensor:
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        if path.suffix == ".npy":
            import numpy as np
            return torch.from_numpy(np.load(path))
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(payload, Mapping):
            for key in ("features", "x", "tensor"):
                if key in payload:
                    payload = payload[key]
                    break
        return torch.as_tensor(payload)
    return torch.as_tensor(value)


class ContractFeatureDataset(Dataset):
    """Read cached temporal features and labels from JSONL."""

    def __init__(self, manifest: str | Path):
        self.path = Path(manifest)
        self.root = self.path.parent
        self.rows = []
        for line_no, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{self.path}:{line_no}: expected a JSON object")
            self.rows.append(row)
        if not self.rows:
            raise ValueError(f"empty feature manifest: {self.path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        features = _load_tensor(row.get("features", row.get("features_path")), self.root).float()
        if features.ndim != 2:
            raise ValueError(f"row {index}: features must have shape [T,D], got {tuple(features.shape)}")
        length = features.shape[0]
        phase = torch.as_tensor(row.get("phase", [0] * length), dtype=torch.long)
        terminal = torch.as_tensor(row.get("terminal", [0] * length), dtype=torch.float32)
        violation = torch.as_tensor(row.get("violation", [0] * length), dtype=torch.float32)
        uncertainty = torch.as_tensor(row.get("uncertainty", 1.0 - terminal), dtype=torch.float32)
        for name, value in (("phase", phase), ("terminal", terminal), ("violation", violation), ("uncertainty", uncertainty)):
            if value.shape[0] != length:
                raise ValueError(f"row {index}: {name} has {value.shape[0]} frames, expected {length}")
        return {"features": features, "phase": phase, "terminal": terminal, "violation": violation, "uncertainty": uncertainty, "id": row.get("id", index)}


def collate_contract_features(batch: list[Mapping[str, Any]]) -> dict[str, Any]:
    max_len = max(item["features"].shape[0] for item in batch)
    dim = batch[0]["features"].shape[-1]
    size = (len(batch), max_len)
    features = torch.zeros(len(batch), max_len, dim, dtype=torch.float32)
    phase = torch.zeros(size, dtype=torch.long)
    terminal = torch.zeros(size)
    violation_width = max(1, max((item["violation"].shape[-1] if item["violation"].ndim > 1 else 1) for item in batch))
    violation = torch.zeros(len(batch), max_len, violation_width)
    uncertainty = torch.ones(size)
    padding_mask = torch.ones(size, dtype=torch.bool)
    ids = []
    for row, item in enumerate(batch):
        length = item["features"].shape[0]
        features[row, :length] = item["features"]
        phase[row, :length] = item["phase"]
        terminal[row, :length] = item["terminal"]
        value = item["violation"]
        value = value[:, None] if value.ndim == 1 else value
        violation[row, :length, :value.shape[-1]] = value
        uncertainty[row, :length] = item["uncertainty"]
        padding_mask[row, :length] = False
        ids.append(item["id"])
    return {"features": features, "phase": phase, "terminal": terminal, "violation": violation, "uncertainty": uncertainty, "padding_mask": padding_mask, "ids": ids}

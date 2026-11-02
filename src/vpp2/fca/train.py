"""Train/evaluate the lightweight FCA temporal head on cached features."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from .data import ContractFeatureDataset, collate_contract_features
from .heads import ContractHead, ContractHeadConfig, contract_loss, save_head


def train_head(
    manifest: str | Path,
    output: str | Path,
    *,
    input_dim: int | None = None,
    hidden_dim: int = 256,
    num_phases: int = 16,
    num_violations: int = 1,
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    valid_fraction: float = 0.1,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    if epochs < 1 or not 0 <= valid_fraction < 1:
        raise ValueError("epochs must be positive and valid_fraction must be in [0, 1)")
    random.seed(seed)
    torch.manual_seed(seed)
    dataset = ContractFeatureDataset(manifest)
    inferred_dim = int(dataset[0]["features"].shape[-1])
    input_dim = inferred_dim if input_dim is None else int(input_dim)
    if input_dim != inferred_dim:
        raise ValueError(f"input_dim={input_dim} does not match cached features ({inferred_dim})")
    valid_size = min(len(dataset) - 1, max(0, int(round(len(dataset) * valid_fraction))))
    train_size = len(dataset) - valid_size
    if valid_size:
        train_set, valid_set = random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(seed))
    else:
        train_set, valid_set = dataset, None
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_contract_features)
    valid_loader = None if valid_set is None else DataLoader(valid_set, batch_size=batch_size, shuffle=False, collate_fn=collate_contract_features)
    model = ContractHead(ContractHeadConfig(input_dim=input_dim, hidden_dim=hidden_dim, num_phases=num_phases, num_violations=num_violations)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(epochs):
        model.train()
        train_values = []
        for batch in loader:
            features = batch["features"].to(device)
            labels = {key: batch[key].to(device) for key in ("phase", "terminal", "violation", "uncertainty")}
            if labels["violation"].shape[-1] == 1 and num_violations > 1:
                labels["violation"] = labels["violation"].expand(-1, -1, num_violations)
            outputs = model(features, batch["padding_mask"].to(device))
            losses = contract_loss(outputs, labels, batch["padding_mask"].to(device))
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_values.append(float(losses["total"].detach().cpu()))
        row = {"epoch": epoch + 1, "train_loss": sum(train_values) / max(len(train_values), 1)}
        if valid_loader is not None:
            model.eval()
            valid_values = []
            with torch.no_grad():
                for batch in valid_loader:
                    outputs = model(batch["features"].to(device), batch["padding_mask"].to(device))
                    labels = {key: batch[key].to(device) for key in ("phase", "terminal", "violation", "uncertainty")}
                    if labels["violation"].shape[-1] == 1 and num_violations > 1:
                        labels["violation"] = labels["violation"].expand(-1, -1, num_violations)
                    valid_values.append(float(contract_loss(outputs, labels, batch["padding_mask"].to(device))["total"].cpu()))
            row["valid_loss"] = sum(valid_values) / max(len(valid_values), 1)
        history.append(row)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_head(model.cpu(), str(output))
    report = {"manifest": str(manifest), "checkpoint": str(output), "train_examples": train_size, "valid_examples": valid_size, "history": history}
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report

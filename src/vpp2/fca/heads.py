"""Trainable temporal contract head for cached VPP2 video features."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class ContractHeadConfig:
    input_dim: int = 1024
    hidden_dim: int = 512
    num_phases: int = 16
    num_violations: int = 8
    dropout: float = 0.1


class ContractHead(nn.Module):
    """Small temporal head; keep the 14B VPP2 backbone frozen for ablations."""
    def __init__(self, config: ContractHeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or ContractHeadConfig()
        self.proj = nn.Sequential(
            nn.LayerNorm(self.config.input_dim),
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.GELU(), nn.Dropout(self.config.dropout),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_dim, nhead=8,
            dim_feedforward=4 * self.config.hidden_dim,
            dropout=self.config.dropout, batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=2)
        self.phase = nn.Linear(self.config.hidden_dim, self.config.num_phases)
        self.terminal = nn.Linear(self.config.hidden_dim, 1)
        self.violation = nn.Linear(self.config.hidden_dim, self.config.num_violations)
        self.uncertainty = nn.Linear(self.config.hidden_dim, 1)

    def forward(self, features: Tensor, padding_mask: Tensor | None = None) -> dict[str, Tensor]:
        if features.ndim != 3:
            raise ValueError(f"features must be [B,T,D], got {tuple(features.shape)}")
        hidden = self.temporal(self.proj(features), src_key_padding_mask=padding_mask)
        return {
            "hidden": hidden,
            "phase_logits": self.phase(hidden),
            "terminal_logits": self.terminal(hidden).squeeze(-1),
            "violation_logits": self.violation(hidden),
            "uncertainty": self.uncertainty(hidden).squeeze(-1).sigmoid(),
        }


def contract_loss(outputs: Mapping[str, Tensor], labels: Mapping[str, Tensor], padding_mask: Tensor | None = None) -> dict[str, Tensor]:
    valid = None if padding_mask is None else ~padding_mask
    phase = F.cross_entropy(outputs["phase_logits"].transpose(1, 2), labels["phase"], reduction="none")
    terminal = F.binary_cross_entropy_with_logits(outputs["terminal_logits"], labels["terminal"].float(), reduction="none")
    violation = F.binary_cross_entropy_with_logits(outputs["violation_logits"], labels["violation"].float(), reduction="none").mean(-1)
    target_uncertainty = labels.get("uncertainty", 1.0 - labels["terminal"].float())
    uncertainty = F.smooth_l1_loss(outputs["uncertainty"], target_uncertainty.float(), reduction="none")
    if valid is not None:
        phase, terminal, violation, uncertainty = [value.masked_select(valid) for value in (phase, terminal, violation, uncertainty)]
    # A fully padded batch can occur when the final distributed shard is empty.
    # Returning a finite zero keeps the training loop synchronized.
    mean = lambda value: value.mean() if value.numel() else outputs["hidden"].sum() * 0.0
    losses = {"phase": mean(phase), "terminal": mean(terminal), "violation": mean(violation), "uncertainty": mean(uncertainty)}
    losses["total"] = losses["phase"] + losses["terminal"] + losses["violation"] + 0.1 * losses["uncertainty"]
    return losses


def save_head(model: ContractHead, path: str) -> None:
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict()}, path)


def load_head(path: str, map_location: str = "cpu") -> ContractHead:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    model = ContractHead(ContractHeadConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"])
    return model

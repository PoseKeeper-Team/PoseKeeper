"""MLP posture classifier for 99-dim MediaPipe Pose vectors."""
from __future__ import annotations

import torch
import torch.nn as nn


class PoseMLP(nn.Module):
    """Simple feed-forward classifier used by realtime inference."""

    def __init__(self, input_dim: int = 99, num_classes: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

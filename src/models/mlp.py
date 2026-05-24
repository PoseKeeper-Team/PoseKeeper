"""MLP posture classifier for 99-dim MediaPipe Pose vectors."""
from __future__ import annotations

import torch
import torch.nn as nn


class PoseMLP(nn.Module):
    """Simple feed-forward classifier used by realtime inference.

    The registered module name is `model` to match weights produced by
    TurtleNeckMLP/train_mlp.py (`model.0.weight`, ...).
    """

    LABELS = {0: "normal", 1: "turtle_neck", 2: "severe_turtle_neck"}

    def __init__(self, input_dim: int = 99, num_classes: int = 3) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    @property
    def net(self) -> nn.Sequential:
        """Backward-compatible alias for older code that accessed `.net`."""
        return self.model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

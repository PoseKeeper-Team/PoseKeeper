from __future__ import annotations

"""Dataset for MLP posture classification."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import config


class PostureDataset(Dataset):
    """MLP dataset backed by `data/processed/mlp/X.npy` and `y.npy`."""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        *,
        x_file: str = "X.npy",
        y_file: str = "y.npy",
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else config.PATHS["processed"] / "mlp"
        self.x_path = self.data_dir / x_file
        self.y_path = self.data_dir / y_file

        self.features = self._load_array(self.x_path).astype(np.float32)
        self.labels = self._load_array(self.y_path).astype(np.int64)
        self._validate()

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        feature = torch.from_numpy(self.features[index])
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return feature, label

    @staticmethod
    def _load_array(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return np.load(path, allow_pickle=False)

    def _validate(self) -> None:
        if self.features.ndim != 2:
            raise ValueError(f"{self.x_path} must have shape (N, 99), got {self.features.shape}")
        if self.features.shape[1] != config.POSE_LANDMARK_DIM:
            raise ValueError(
                f"{self.x_path} feature dim mismatch: "
                f"{self.features.shape[1]} != {config.POSE_LANDMARK_DIM}"
            )
        if self.labels.ndim != 1:
            raise ValueError(f"{self.y_path} must have shape (N,), got {self.labels.shape}")
        if len(self.features) != len(self.labels):
            raise ValueError(
                f"X/y sample count mismatch: {len(self.features)} != {len(self.labels)}"
            )


def load_posture_dataset(data_dir: str | Path | None = None) -> PostureDataset:
    return PostureDataset(data_dir=data_dir)

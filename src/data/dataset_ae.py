from __future__ import annotations

"""Dataset for Autoencoder posture reconstruction."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import config


class AutoencoderDataset(Dataset):
    """AE dataset returning `(feature, feature)` for reconstruction training."""

    _SPLIT_FILES = {
        "all": ("ae", "normal_poses.npy"),
        "train": ("autoencoder", "X_train.npy"),
        "val": ("autoencoder", "X_val.npy"),
    }

    def __init__(
        self,
        data_dir: str | Path | None = None,
        *,
        split: str = "train",
        data_file: str | None = None,
    ) -> None:
        if split not in self._SPLIT_FILES:
            raise ValueError("split must be one of 'train', 'val', 'all'")

        default_subdir, default_file = self._SPLIT_FILES[split]
        self.data_dir = (
            Path(data_dir)
            if data_dir is not None
            else config.PATHS["processed"] / default_subdir
        )
        self.data_path = self.data_dir / (data_file or default_file)
        self.split = split

        self.features = self._load_array(self.data_path).astype(np.float32)
        self._validate()

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        feature = torch.from_numpy(self.features[index])
        return feature, feature

    @staticmethod
    def _load_array(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return np.load(path, allow_pickle=False)

    def _validate(self) -> None:
        if self.features.ndim != 2:
            raise ValueError(f"{self.data_path} must have shape (N, 99), got {self.features.shape}")
        if self.features.shape[1] != config.POSE_LANDMARK_DIM:
            raise ValueError(
                f"{self.data_path} feature dim mismatch: "
                f"{self.features.shape[1]} != {config.POSE_LANDMARK_DIM}"
            )


def load_autoencoder_dataset(
    data_dir: str | Path | None = None,
    *,
    split: str = "train",
) -> AutoencoderDataset:
    return AutoencoderDataset(data_dir=data_dir, split=split)

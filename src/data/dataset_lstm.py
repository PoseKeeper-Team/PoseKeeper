from __future__ import annotations

"""Dataset for LSTM focus sequence classification."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import config


class FocusSequenceDataset(Dataset):
    """LSTM dataset backed by `data/processed/lstm/X.npy` and `y.npy`."""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        *,
        x_file: str = "X.npy",
        y_file: str = "y.npy",
        expected_seq_len: int | None = None,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else config.PATHS["processed"] / "lstm"
        self.x_path = self.data_dir / x_file
        self.y_path = self.data_dir / y_file
        self.expected_seq_len = expected_seq_len

        self.sequences = self._load_array(self.x_path).astype(np.float32)
        self.labels = self._load_array(self.y_path).astype(np.int64)
        self._validate()

    def __len__(self) -> int:
        return int(self.sequences.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = torch.from_numpy(self.sequences[index])
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return sequence, label

    @staticmethod
    def _load_array(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return np.load(path, allow_pickle=False)

    def _validate(self) -> None:
        if self.sequences.ndim != 3:
            raise ValueError(
                f"{self.x_path} must have shape (N, T, {config.LSTM_FEATURE_DIM}), "
                f"got {self.sequences.shape}"
            )
        if self.sequences.shape[2] != config.LSTM_FEATURE_DIM:
            raise ValueError(
                f"{self.x_path} feature dim mismatch: "
                f"{self.sequences.shape[2]} != {config.LSTM_FEATURE_DIM}"
            )
        if self.expected_seq_len is not None and self.sequences.shape[1] != self.expected_seq_len:
            raise ValueError(
                f"{self.x_path} sequence length mismatch: "
                f"{self.sequences.shape[1]} != {self.expected_seq_len}"
            )
        if self.labels.ndim != 1:
            raise ValueError(f"{self.y_path} must have shape (N,), got {self.labels.shape}")
        if len(self.sequences) != len(self.labels):
            raise ValueError(
                f"X/y sample count mismatch: {len(self.sequences)} != {len(self.labels)}"
            )


def load_focus_sequence_dataset(
    data_dir: str | Path | None = None,
    *,
    expected_seq_len: int | None = None,
) -> FocusSequenceDataset:
    return FocusSequenceDataset(data_dir=data_dir, expected_seq_len=expected_seq_len)

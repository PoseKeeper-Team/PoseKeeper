from __future__ import annotations

"""PyTorch Dataset entrypoints for MLP / LSTM / Autoencoder."""

from pathlib import Path
from typing import Literal

from torch.utils.data import Dataset

from src.data.dataset_ae import AutoencoderDataset, load_autoencoder_dataset
from src.data.dataset_lstm import FocusSequenceDataset, load_focus_sequence_dataset
from src.data.dataset_mlp import PostureDataset, load_posture_dataset

DatasetName = Literal["mlp", "lstm", "ae", "autoencoder"]


def create_dataset(
    name: DatasetName,
    *,
    data_dir: str | Path | None = None,
    split: str = "train",
    expected_seq_len: int | None = None,
) -> Dataset:
    """Create a model-specific Dataset with one stable helper.

    Args:
        name: "mlp", "lstm", "ae", or "autoencoder".
        data_dir: Optional override for the processed data directory.
        split: AE split, one of "train", "val", "all". Ignored by MLP/LSTM.
        expected_seq_len: Optional LSTM sequence length validation.
    """
    if name == "mlp":
        return load_posture_dataset(data_dir=data_dir)
    if name == "lstm":
        return load_focus_sequence_dataset(
            data_dir=data_dir,
            expected_seq_len=expected_seq_len,
        )
    if name in {"ae", "autoencoder"}:
        return load_autoencoder_dataset(data_dir=data_dir, split=split)
    raise ValueError("name must be one of 'mlp', 'lstm', 'ae', 'autoencoder'")


__all__ = [
    "AutoencoderDataset",
    "FocusSequenceDataset",
    "PostureDataset",
    "create_dataset",
    "load_autoencoder_dataset",
    "load_focus_sequence_dataset",
    "load_posture_dataset",
]

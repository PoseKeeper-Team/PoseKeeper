"""LSTM(집중도) 학습 스크립트. 학습 후 weights/lstm.pth 저장."""
from __future__ import annotations

"""LSTM 집중도 분류 모델 학습 스크립트.

입력:
    data/processed/lstm/X.npy
    data/processed/lstm/y.npy

출력:
    weights/lstm.pth

실행:
    python -m src.train.train_lstm --epochs 30 --batch-size 32
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import config
from src.data.dataset_lstm import FocusSequenceDataset
from src.models.lstm import PoseLSTM


def seed_everything(seed: int = 42) -> None:
    """학습 결과가 매번 너무 크게 달라지지 않도록 난수 고정."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_stratified_split(
    labels: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """
    train / validation 데이터를 라벨별로 나눈다.

    그냥 전체를 랜덤으로 나누면 특정 라벨이 validation에 거의 안 들어갈 수 있다.
    그래서 normal / drowsy / distracted 각각에서 일정 비율을 validation으로 보낸다.
    """
    rng = np.random.default_rng(seed)

    train_indices: list[int] = []
    val_indices: list[int] = []

    for class_id in sorted(np.unique(labels)):
        class_indices = np.where(labels == class_id)[0]
        rng.shuffle(class_indices)

        # 클래스 샘플이 1개뿐이면 validation으로 빼면 train에 해당 클래스가 사라진다.
        if len(class_indices) <= 1:
            train_indices.extend(class_indices.tolist())
            continue

        val_count = max(1, int(len(class_indices) * val_ratio))
        val_count = min(val_count, len(class_indices) - 1)

        val_indices.extend(class_indices[:val_count].tolist())
        train_indices.extend(class_indices[val_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    if not train_indices:
        raise RuntimeError("train 데이터가 없습니다. 전처리 데이터 개수를 확인하세요.")

    if not val_indices:
        raise RuntimeError("validation 데이터가 없습니다. 데이터 수를 늘리거나 --val-ratio를 조정하세요.")

    return train_indices, val_indices


def make_class_weights(labels: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    """
    클래스 불균형 보정용 가중치.

    예를 들어 normal 데이터가 많고 drowsy 데이터가 적으면,
    drowsy를 틀렸을 때 loss가 더 크게 반영되도록 한다.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)

    # 0개인 클래스가 있으면 division by zero 방지.
    # 다만 실제로는 0개 클래스가 있으면 학습 데이터 자체를 보강하는 게 맞다.
    counts[counts == 0] = 1.0

    weights = counts.sum() / (num_classes * counts)

    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(X)
        loss = criterion(logits, y)

        loss.backward()

        # LSTM은 gradient가 튈 수 있어서 clipping을 넣는다.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        pred = logits.argmax(dim=1)

        total_loss += loss.item() * y.size(0)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)

    return avg_loss, acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray]:
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    confusion = np.zeros((3, 3), dtype=np.int64)

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        loss = criterion(logits, y)

        pred = logits.argmax(dim=1)

        total_loss += loss.item() * y.size(0)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))

        for true_label, pred_label in zip(y.cpu().numpy(), pred.cpu().numpy()):
            confusion[int(true_label), int(pred_label)] += 1

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)

    return avg_loss, acc, confusion


def train_lstm(
    data_dir: Path,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    seed_everything(seed)

    dataset = FocusSequenceDataset(
        data_dir=data_dir,
        expected_seq_len=config.LSTM_SEQUENCE_LENGTH,
    )

    labels = dataset.labels

    train_idx, val_idx = make_stratified_split(
        labels=labels,
        val_ratio=val_ratio,
        seed=seed,
    )

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    device = torch.device(config.DEVICE)

    model = PoseLSTM(
        input_dim=config.LSTM_FEATURE_DIM,
        num_classes=3,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    class_weights = make_class_weights(
        labels=labels[train_idx],
        num_classes=3,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    config.PATHS["weights"].mkdir(parents=True, exist_ok=True)
    save_path = config.WEIGHT_FILES["lstm"]

    best_val_acc = -1.0
    best_confusion = None

    print()
    print("[LSTM 학습 시작]")
    print(f"device        : {device}")
    print(f"data_dir      : {data_dir}")
    print(f"X shape       : {dataset.sequences.shape}")
    print(f"y shape       : {dataset.labels.shape}")
    print(f"train samples : {len(train_dataset)}")
    print(f"val samples   : {len(val_dataset)}")
    print(f"class counts  : {np.bincount(labels, minlength=3).tolist()}")
    print(f"class weights : {class_weights.detach().cpu().numpy().round(4).tolist()}")
    print(f"save path     : {save_path}")
    print()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc, confusion = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
        )

        if epoch % 5 == 0 or epoch == epochs:
            print("confusion matrix [true rows x pred cols]")
            print(confusion)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_confusion = confusion.copy()

            # 중요:
            # predictor.py는 PoseLSTM을 만든 뒤 load_state_dict()로 바로 불러온다.
            # 그래서 checkpoint dict가 아니라 state_dict만 저장해야 한다.
            torch.save(model.state_dict(), save_path)

            print(f"[SAVE] best model saved: {save_path}")
            print("best confusion matrix [true rows x pred cols]")
            print(best_confusion)

    print()
    print(f"[LSTM 학습 완료] best_val_acc={best_val_acc:.3f}")
    if best_confusion is not None:
        print("best confusion matrix [true rows x pred cols]")
        print(best_confusion)
    print(f"[저장 파일] {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PoseKeeper LSTM focus classifier")

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=config.PATHS["processed"] / "lstm",
        help="X.npy, y.npy가 있는 폴더",
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train_lstm(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
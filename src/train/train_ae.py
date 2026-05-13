"""Autoencoder(이상자세) 학습 스크립트. 정상 자세 데이터만으로 비지도 학습."""


"""
train.py
---------
Autoencoder 학습 스크립트.
수집한 정상 자세 .npy 파일로 모델을 학습하고,
학습 후 이상 탐지 임계값(threshold)을 자동 계산하여 함께 저장한다.

사용법 (VS Code 터미널 / Google Colab):
    python train.py --data data/normal_poses.npy --epochs 100

Google Colab 예시:
    !python train.py --data data/normal_poses.npy --epochs 150 --latent 16
"""

from config import PATHS, WEIGHT_FILES
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

from src.models.autoencoder import PoseAutoencoder


# ─────────────────────────────────────────────
# 학습 함수
# ─────────────────────────────────────────────

def train(
    data_path: str,
    save_dir: str   = "checkpoints",
    epochs: int     = 100,
    batch_size: int = 64,
    lr: float       = 1e-3,
    latent_dim: int = 16,
    val_ratio: float= 0.1,
    threshold_percentile: float = 95.0,
):
    """
    Args:
        data_path            : 정상 자세 .npy 파일 경로
        save_dir             : 모델 및 임계값 저장 폴더
        epochs               : 학습 에폭 수
        batch_size           : 배치 크기
        lr                   : 학습률
        latent_dim           : 잠재 공간 차원
        val_ratio            : 검증 셋 비율
        threshold_percentile : 임계값 계산에 사용할 백분위수 (기본 95%)
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[장치] {device}")

    # ── 데이터 로드 ──────────────────────────────────────────────
    raw = np.load(data_path).astype(np.float32)
    print(f"[데이터] {data_path}  shape: {raw.shape}")

    tensor = torch.tensor(raw)
    dataset = TensorDataset(tensor)

    n_val  = max(1, int(len(dataset) * val_ratio))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                       generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)
    print(f"[분할] 학습 {n_train}개 | 검증 {n_val}개")

    # ── 모델 / 옵티마이저 / 손실 ─────────────────────────────────
    model = PoseAutoencoder(input_dim=99, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    # ── 학습 루프 ────────────────────────────────────────────────
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        running_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out  = model(batch)
            loss = criterion(out, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(batch)
        train_loss = running_loss / n_train

        # Validation
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                out  = model(batch)
                loss = criterion(out, batch)
                val_running += loss.item() * len(batch)
        val_loss = val_running / n_val

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "autoencoder_best.pth"))

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{epochs}  train_loss: {train_loss:.6f}  val_loss: {val_loss:.6f}")

    print(f"\n[최적 검증 손실] {best_val_loss:.6f}")

    # ── 임계값 계산 ──────────────────────────────────────────────
    # 학습 전체 데이터에 대한 재구성 오차 분포에서 percentile을 임계값으로 설정
    model.load_state_dict(torch.load(os.path.join(save_dir, "autoencoder_best.pth"), map_location=device))
    model.eval()

    all_errors = []
    full_loader = DataLoader(TensorDataset(tensor), batch_size=256, shuffle=False)
    with torch.no_grad():
        for (batch,) in full_loader:
            batch = batch.to(device)
            err = model.reconstruction_error(batch)
            all_errors.extend(err.cpu().numpy().tolist())

    all_errors = np.array(all_errors)
    threshold = float(np.percentile(all_errors, threshold_percentile))
    print(f"[임계값] {threshold_percentile}th 백분위 → {threshold:.6f}")

    # 임계값 저장
    threshold_path = os.path.join(save_dir, "threshold.npy")
    np.save(threshold_path, np.array([threshold]))
    print(f"[저장] 임계값 → {threshold_path}")

    # ── 손실 그래프 ──────────────────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Autoencoder Training Loss")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"[저장] 손실 그래프 → {plot_path}")

    # ── 재구성 오차 분포 그래프 ──────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.hist(all_errors, bins=60, color="steelblue", edgecolor="white", alpha=0.8)
    plt.axvline(threshold, color="red", linewidth=2, label=f"Threshold ({threshold_percentile}th pct) = {threshold:.4f}")
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Frequency")
    plt.title("정상 자세 재구성 오차 분포")
    plt.legend()
    plt.tight_layout()
    dist_path = os.path.join(save_dir, "error_distribution.png")
    plt.savefig(dist_path)
    plt.show()
    print(f"[저장] 오차 분포 그래프 → {dist_path}")

    return threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoencoder 학습")
    parser.add_argument("--data",       default=str(PATHS["raw"] / "normal_poses.npy"), help="학습 데이터 경로")
    parser.add_argument("--save_dir",   default=str(PATHS["weights"]),           help="모델 저장 폴더")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--latent",     type=int,   default=16,          help="잠재 공간 차원")
    parser.add_argument("--percentile", type=float, default=95.0,        help="임계값 백분위")
    args = parser.parse_args()

    train(
        data_path=args.data,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent,
        threshold_percentile=args.percentile,
    )

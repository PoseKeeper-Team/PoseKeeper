"""Autoencoder(이상자세) 학습 스크립트. 정상 자세 데이터만으로 비지도 학습."""

from config import PATHS, WEIGHT_FILES
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt

from src.models.autoencoder import PoseAutoencoder
from src.data.dataset_ae import AutoencoderDataset


# ─────────────────────────────────────────────
# 학습 함수
# ─────────────────────────────────────────────

def train(
    save_dir: str = "checkpoints",
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    latent_dim: int = 16,
    threshold_percentile: float = 95.0,
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[장치] {device}")

    # ── 데이터 로드 (AutoencoderDataset 사용) ────────────────────
    train_set = AutoencoderDataset(split="train")
    val_set   = AutoencoderDataset(split="val")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)

    n_train = len(train_set)
    n_val   = len(val_set)
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
        for batch, _ in train_loader:  # (feature, feature) 반환
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
            for batch, _ in val_loader:  # (feature, feature) 반환
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

    # ── 임계값 계산 (train + val 전체 데이터 기준) ───────────────
    model.load_state_dict(torch.load(os.path.join(save_dir, "autoencoder_best.pth"), map_location=device))
    model.eval()

    full_set    = ConcatDataset([train_set, val_set])
    full_loader = DataLoader(full_set, batch_size=256, shuffle=False)

    all_errors = []
    with torch.no_grad():
        for batch, _ in full_loader:
            batch = batch.to(device)
            err = model.reconstruction_error(batch)
            all_errors.extend(err.cpu().numpy().tolist())

    all_errors = np.array(all_errors)
    threshold = float(np.percentile(all_errors, threshold_percentile))
    print(f"[임계값] {threshold_percentile}th 백분위 → {threshold:.6f}")

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
    plt.axvline(threshold, color="red", linewidth=2,
                label=f"Threshold ({threshold_percentile}th pct) = {threshold:.4f}")
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
    import argparse
    parser = argparse.ArgumentParser(description="Autoencoder 학습")
    parser.add_argument("--save_dir",   default=str(PATHS["weights"]), help="모델 저장 폴더")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--latent",     type=int,   default=16,   help="잠재 공간 차원")
    parser.add_argument("--percentile", type=float, default=95.0, help="임계값 백분위")
    args = parser.parse_args()

    train(
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent,
        threshold_percentile=args.percentile,
    )
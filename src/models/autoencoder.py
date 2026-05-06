"""이상 자세 탐지 Autoencoder — 정상 자세 99차원만으로 비지도 학습.

Owner: 김병훈. 학습 스크립트는 src/train/train_ae.py.
"""
"""
model.py
---------
MediaPipe Pose 랜드마크(99차원) 기반 Autoencoder 모델 정의.
정상 자세 데이터만으로 학습하며, 재구성 오차(Reconstruction Error)로
이상 자세를 비지도 학습 방식으로 탐지한다.
"""

import torch
import torch.nn as nn


class PoseAutoencoder(nn.Module):
    """
    입력: MediaPipe Pose 랜드마크 좌표 (x, y, z) × 33개 = 99차원 벡터
    구조: Encoder → 저차원 잠재 벡터(latent) → Decoder → 재구성 벡터
    """

    def __init__(self, input_dim: int = 99, latent_dim: int = 16):
        """
        Args:
            input_dim  : 입력 차원 (기본값 99 = 33 landmarks × 3)
            latent_dim : 잠재 공간 차원 (압축 크기)
        """
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, latent_dim),
            nn.ReLU(),
        )

        # ── Decoder ──────────────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, input_dim),
            # 출력 활성화 없음: 랜드마크 좌표는 실수 범위가 다양함
        )

    def forward(self, x: torch.Tensor):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """잠재 벡터만 반환 (시각화·분석용)"""
        return self.encoder(x)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        샘플별 재구성 오차(MSE) 반환.
        Returns:
            shape (batch,) 텐서 — 값이 클수록 이상 자세일 가능성 높음
        """
        with torch.no_grad():
            x_hat = self.forward(x)
            # 각 샘플의 평균 제곱 오차
            error = ((x - x_hat) ** 2).mean(dim=1)
        return error


if __name__ == "__main__":
    # 간단한 동작 확인
    model = PoseAutoencoder(input_dim=99, latent_dim=16)
    dummy = torch.randn(8, 99)          # 배치 크기 8
    out   = model(dummy)
    err   = model.reconstruction_error(dummy)
    print(f"입력 shape : {dummy.shape}")
    print(f"출력 shape : {out.shape}")
    print(f"재구성 오차: {err}")
    print("모델 구조:\n", model)

"""실시간 추론 통합 모듈.

3개 모델(MLP/LSTM/AE)을 동시에 lazy-load 해서 한 번의 MediaPipe 처리 결과를
공유하도록 구성. config.BG_* 값을 따라가며, 대시보드 모드로 전환되면
config.DASHBOARD_* 로 스위치.
"""

"""
detector.py
------------
학습된 Autoencoder로 실시간 이상 자세를 탐지한다.
다른 모듈(MLP 거북목, LSTM 집중도)과 연동할 수 있도록
AnomalyDetector 클래스를 제공한다.

단독 실행 시 웹캠 데모:
    python detector.py \
        --model checkpoints/autoencoder_best.pth \
        --threshold checkpoints/threshold.npy
"""

import time
from collections import deque

import cv2
import numpy as np
import torch

from src.models.autoencoder import PoseAutoencoder
from src.utils.mediapipe_utils import extract_landmarks, normalize_landmarks, draw_landmarks, mp_pose


# ─────────────────────────────────────────────
# AnomalyDetector 클래스 (다른 모듈에서 import 가능)
# ─────────────────────────────────────────────

class AnomalyDetector:
    """
    실시간 이상 자세 탐지기.

    다른 모듈에서 사용 예시:
        from detector import AnomalyDetector
        detector = AnomalyDetector("checkpoints/autoencoder_best.pth",
                                   "checkpoints/threshold.npy")
        is_anomaly, error, level = detector.predict(landmarks_vector)
    """

    LEVEL_LABELS = {0: "정상", 1: "주의", 2: "경고"}
    LEVEL_COLORS_BGR = {0: (0, 220, 0), 1: (0, 165, 255), 2: (0, 0, 220)}

    def __init__(
        self,
        model_path: str,
        threshold_path: str,
        input_dim: int  = 99,
        latent_dim: int = 16,
        smooth_window: int = 10,   # 오차 스무딩 윈도우 크기
    ):
        """
        Args:
            model_path     : 학습된 모델 가중치 경로 (.pth)
            threshold_path : 저장된 임계값 경로 (.npy)
            smooth_window  : 시계열 스무딩 윈도우 (프레임)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 모델 로드
        self.model = PoseAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # 임계값 로드
        self.threshold = float(np.load(threshold_path)[0])

        # 스무딩 버퍼
        self._error_buffer = deque(maxlen=smooth_window)

        print(f"[AnomalyDetector] 모델: {model_path}")
        print(f"[AnomalyDetector] 임계값: {self.threshold:.6f}")

    def predict(self, landmark_vector: np.ndarray) -> tuple[bool, float, int]:
        """
        하나의 랜드마크 벡터에 대해 이상 여부를 판단한다.

        Args:
            landmark_vector: shape (99,) — normalize_landmarks() 적용 후 값

        Returns:
            is_anomaly (bool)  : 이상 자세 여부
            smoothed_error (float): 스무딩된 재구성 오차
            level (int)        : 0=정상, 1=주의(임계값 1~1.5×), 2=경고(1.5×+)
        """
        x = torch.tensor(landmark_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        raw_error = float(self.model.reconstruction_error(x).item())

        self._error_buffer.append(raw_error)
        smoothed = float(np.mean(self._error_buffer))

        # 위험 수준 분류
        if smoothed < self.threshold:
            level = 0
        elif smoothed < self.threshold * 1.5:
            level = 1
        else:
            level = 2

        return level > 0, smoothed, level

    def reset(self):
        """스무딩 버퍼 초기화 (세션 재시작 시 호출)"""
        self._error_buffer.clear()

    @property
    def current_error(self) -> float:
        """현재 스무딩된 오차값"""
        return float(np.mean(self._error_buffer)) if self._error_buffer else 0.0
"""Project-wide configuration.

Background mode prioritizes CPU/RAM frugality (the app sits in the tray for
hours). Dashboard mode prioritizes visual clarity. Training uses its own
resolution. All three are kept separate so a single knob change can't
accidentally bloat the background path.
"""
from __future__ import annotations

from pathlib import Path

from src.utils.device import detect_device, is_colab


PROJECT_ROOT = Path(__file__).resolve().parent

DEVICE: str = detect_device()      # "cuda" | "mps" | "cpu"
IS_COLAB: bool = is_colab()


# --- Background tray mode (default) -----------------------------------
# 30 fps webcam -> 6 fps inference. MediaPipe Pose Lite is the cheapest
# option that still produces usable landmarks.
BG_FRAME_SKIP: int = 5
BG_CAPTURE_RESOLUTION: tuple[int, int] = (320, 240)
BG_MEDIAPIPE_COMPLEXITY: int = 0
BG_FACEMESH_REFINE: bool = False

# --- Dashboard mode (only when user opens the window) -----------------
DASHBOARD_FRAME_SKIP: int = 2
DASHBOARD_CAPTURE_RESOLUTION: tuple[int, int] = (640, 480)
DASHBOARD_MEDIAPIPE_COMPLEXITY: int = 1
DASHBOARD_FACEMESH_REFINE: bool = True

# --- Training data collection -----------------------------------------
TRAIN_CAPTURE_RESOLUTION: tuple[int, int] = (640, 480)


# --- Alerts -----------------------------------------------------------
ALERT_SENSITIVITY: str = "medium"           # "low" | "medium" | "high"
TURTLE_NECK_THRESHOLD_SEC: float = 5.0
DROWSINESS_THRESHOLD_SEC: float = 3.0
DISTRACTION_THRESHOLD_SEC: float = 10.0
ALERT_COOLDOWN_SEC: float = 60.0
EAR_THRESHOLD: float = 0.20
MAR_THRESHOLD: float = 0.60
HEADPOSE_YAW_THRESHOLD_DEG: float = 35.0
HEADPOSE_PITCH_THRESHOLD_DEG: float = 25.0


# --- Model input dims (locked by MediaPipe) ---------------------------
POSE_LANDMARK_DIM: int = 33 * 3              # MLP / Autoencoder input
FACEMESH_LANDMARK_COUNT: int = 468           # LSTM raw landmark count
LSTM_SEQUENCE_LENGTH: int = 30               # ~5 seconds at 6 fps
LSTM_FEATURE_DIM: int = FACEMESH_LANDMARK_COUNT * 3 + 1 + 1 + 3


# --- Paths ------------------------------------------------------------
PATHS: dict[str, Path] = {
    "weights": PROJECT_ROOT / "weights",
    "raw": PROJECT_ROOT / "data" / "raw",
    "processed": PROJECT_ROOT / "data" / "processed",
    "db": PROJECT_ROOT / "data" / "posture.db",
}

WEIGHT_FILES: dict[str, Path] = {
    "mlp": PATHS["weights"] / "mlp.pth",
    "lstm": PATHS["weights"] / "lstm.pth",
    "autoencoder": PATHS["weights"] / "autoencoder_best.pth",
    "autoencoder_threshold": PATHS["weights"] / "threshold.npy",
}

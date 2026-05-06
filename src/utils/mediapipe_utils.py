"""MediaPipe Pose / FaceMesh wrapper + derived features.

Public API
----------
should_process(mode)                        -> bool
extract_pose_landmarks(frame, mode)         -> ndarray(99,)   | None
process_face(frame, mode)                   -> NormalizedLandmarkList | None
extract_face_landmarks(frame, mode)         -> ndarray(1404,) | None
compute_ear(face_lm)                        -> float
compute_mar(face_lm)                        -> float
estimate_head_pose(face_lm, image_size)     -> (yaw, pitch, roll)
build_lstm_feature(face_lm, image_size)     -> ndarray(1409,) | None

Note on detection cost
----------------------
process_face() runs FaceMesh once and returns the raw landmark list.
Callers that need both extract_face_landmarks() and build_lstm_feature()
should call process_face() first and reuse the result to avoid running
the detector twice on the same frame.
"""
from __future__ import annotations

import logging

import cv2
import mediapipe as mp
import numpy as np

import config

logger = logging.getLogger(__name__)

_VALID_MODES = frozenset({"bg", "dashboard"})

# ---------------------------------------------------------------------------
# Singleton detectors — one instance per mode avoids repeated model loading
# ---------------------------------------------------------------------------

_pose_detectors: dict[str, mp.solutions.pose.Pose | None] = {
    "bg": None,
    "dashboard": None,
}
_face_detectors: dict[str, mp.solutions.face_mesh.FaceMesh | None] = {
    "bg": None,
    "dashboard": None,
}


def _check_mode(mode: str) -> None:
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be 'bg' or 'dashboard', got {mode!r}")


def _get_pose(mode: str) -> mp.solutions.pose.Pose:
    if _pose_detectors[mode] is None:
        complexity = (
            config.BG_MEDIAPIPE_COMPLEXITY
            if mode == "bg"
            else config.DASHBOARD_MEDIAPIPE_COMPLEXITY
        )
        _pose_detectors[mode] = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=complexity,
            enable_segmentation=False,
        )
    return _pose_detectors[mode]  # type: ignore[return-value]


def _get_face(mode: str) -> mp.solutions.face_mesh.FaceMesh:
    if _face_detectors[mode] is None:
        refine = (
            config.BG_FACEMESH_REFINE
            if mode == "bg"
            else config.DASHBOARD_FACEMESH_REFINE
        )
        _face_detectors[mode] = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=refine,
        )
    return _face_detectors[mode]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Frame skip
# ---------------------------------------------------------------------------

_frame_counter: dict[str, int] = {"bg": 0, "dashboard": 0}


def should_process(mode: str = "bg") -> bool:
    """Return True on frames that pass the FRAME_SKIP gate for the given mode.

    Args:
        mode: "bg" or "dashboard".

    Returns:
        True every (FRAME_SKIP + 1)-th call; False otherwise.

    Raises:
        ValueError: if mode is not "bg" or "dashboard".
    """
    _check_mode(mode)
    skip = config.BG_FRAME_SKIP if mode == "bg" else config.DASHBOARD_FRAME_SKIP
    _frame_counter[mode] = (_frame_counter[mode] + 1) % (skip + 1)
    return _frame_counter[mode] == 0


# ---------------------------------------------------------------------------
# Landmark extraction
# ---------------------------------------------------------------------------

def extract_pose_landmarks(
    frame: np.ndarray | None,
    mode: str = "bg",
) -> np.ndarray | None:
    """Extract 33 Pose landmarks from a BGR frame and flatten to (99,).

    Args:
        frame: BGR numpy array from webcam.
        mode: "bg" or "dashboard".

    Returns:
        float32 ndarray of shape (POSE_LANDMARK_DIM,) = (99,),
        or None if frame is None or landmarks not detected.

    Raises:
        ValueError: if mode is invalid.
    """
    _check_mode(mode)
    if frame is None:
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = _get_pose(mode).process(frame_rgb)
    if not results.pose_landmarks:
        return None
    lm = results.pose_landmarks.landmark
    return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32).flatten()


def process_face(
    frame: np.ndarray | None,
    mode: str = "bg",
):
    """Run FaceMesh detection and return the first face landmark list.

    Use this when the same frame is passed to multiple downstream functions
    (compute_ear, compute_mar, estimate_head_pose, build_lstm_feature) so
    the detector runs only once.

    Args:
        frame: BGR numpy array from webcam.
        mode: "bg" or "dashboard".

    Returns:
        MediaPipe NormalizedLandmarkList for the first detected face,
        or None if frame is None or no face is detected.

    Raises:
        ValueError: if mode is invalid.
    """
    _check_mode(mode)
    if frame is None:
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = _get_face(mode).process(frame_rgb)
    if not results.multi_face_landmarks:
        return None
    return results.multi_face_landmarks[0]


def extract_face_landmarks(
    frame: np.ndarray | None,
    mode: str = "bg",
) -> np.ndarray | None:
    """Extract 468 FaceMesh landmarks from a BGR frame and flatten to (1404,).

    Args:
        frame: BGR numpy array from webcam.
        mode: "bg" or "dashboard".

    Returns:
        float32 ndarray of shape (FACEMESH_LANDMARK_COUNT * 3,) = (1404,),
        or None if not detected.

    Raises:
        ValueError: if mode is invalid.
    """
    face_lm = process_face(frame, mode)
    if face_lm is None:
        return None
    lm = face_lm.landmark[: config.FACEMESH_LANDMARK_COUNT]
    return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32).flatten()


# ---------------------------------------------------------------------------
# Derived features — helpers
# ---------------------------------------------------------------------------

# EAR landmark indices (P1..P6 per eye)
# P1=inner corner, P4=outer corner (horizontal)
# P2,P3,P5,P6 = vertical pairs
_LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
_RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# MAR landmark indices
# A/G, B/F, C/E = vertical pairs; D/H = horizontal corners
_MAR_A, _MAR_G = 82, 87    # upper-left / lower-left
_MAR_B, _MAR_F = 13, 14    # upper-center / lower-center
_MAR_C, _MAR_E = 312, 317  # upper-right / lower-right
_MAR_D, _MAR_H = 78, 308   # left corner / right corner

# 3-D canonical face model points for solvePnP (mm, nose-tip = origin)
_FACE_3D = np.array(
    [
        [0.0, 0.0, 0.0],           # 1   nose tip
        [0.0, -330.0, -65.0],      # 152 chin
        [-225.0, 170.0, -135.0],   # 33  left eye outer corner
        [225.0, 170.0, -135.0],    # 263 right eye outer corner
        [-150.0, -150.0, -125.0],  # 78  left mouth corner
        [150.0, -150.0, -125.0],   # 308 right mouth corner
    ],
    dtype=np.float64,
)
_FACE_PNP_IDX = [1, 152, 33, 263, 78, 308]

_headpose_failures: int = 0


def _pt(face_lm, idx: int) -> np.ndarray:
    p = face_lm.landmark[idx]
    return np.array([p.x, p.y], dtype=np.float64)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _ear_one(face_lm, idx: list[int]) -> float:
    p1, p2, p3, p4, p5, p6 = (_pt(face_lm, i) for i in idx)
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * _dist(p1, p4) + 1e-6)


# ---------------------------------------------------------------------------
# Derived features — public
# ---------------------------------------------------------------------------

def compute_ear(face_lm) -> float:
    """Eye Aspect Ratio averaged over both eyes.

    EAR = (||P2-P6|| + ||P3-P5||) / (2 × ||P1-P4||) per eye.

    Args:
        face_lm: MediaPipe NormalizedLandmarkList (multi_face_landmarks[0]).

    Returns:
        EAR float in roughly [0.0, 0.5].
        Values below config.EAR_THRESHOLD indicate eye closure.
    """
    return (_ear_one(face_lm, _LEFT_EYE_IDX) + _ear_one(face_lm, _RIGHT_EYE_IDX)) / 2.0


def compute_mar(face_lm) -> float:
    """Mouth Aspect Ratio for yawn detection.

    MAR = (||A-G|| + ||B-F|| + ||C-E||) / (2 × ||D-H||)

    Args:
        face_lm: MediaPipe NormalizedLandmarkList.

    Returns:
        MAR float. Values above config.MAR_THRESHOLD indicate open mouth.
    """
    A, G = _pt(face_lm, _MAR_A), _pt(face_lm, _MAR_G)
    B, F = _pt(face_lm, _MAR_B), _pt(face_lm, _MAR_F)
    C, E = _pt(face_lm, _MAR_C), _pt(face_lm, _MAR_E)
    D, H = _pt(face_lm, _MAR_D), _pt(face_lm, _MAR_H)
    return (
        _dist(A, G) + _dist(B, F) + _dist(C, E)
    ) / (2.0 * _dist(D, H) + 1e-6)


def estimate_head_pose(
    face_lm,
    image_size: tuple[int, int],
) -> tuple[float, float, float]:
    """Estimate yaw, pitch, roll in degrees from FaceMesh landmarks via solvePnP.

    Args:
        face_lm: MediaPipe NormalizedLandmarkList.
        image_size: (width, height) of the source frame in pixels.

    Returns:
        (yaw_deg, pitch_deg, roll_deg).
        Returns (0.0, 0.0, 0.0) on solvePnP failure and logs a debug message.
    """
    global _headpose_failures
    w, h = image_size
    pts_2d = np.array(
        [
            [face_lm.landmark[i].x * w, face_lm.landmark[i].y * h]
            for i in _FACE_PNP_IDX
        ],
        dtype=np.float64,
    )
    focal = float(w)
    cam = np.array(
        [[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, _ = cv2.solvePnP(
        _FACE_3D, pts_2d, cam, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        _headpose_failures += 1
        logger.debug("solvePnP failed (total failures: %d)", _headpose_failures)
        return (0.0, 0.0, 0.0)

    rmat, _ = cv2.Rodrigues(rvec)
    sy = float(np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2))
    if sy > 1e-6:
        roll  = float(np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0])))
        pitch = float(np.degrees(np.arctan2(-rmat[2, 0], sy)))
        yaw   = float(np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2])))
    else:
        roll  = float(np.degrees(np.arctan2(-rmat[0, 1], rmat[1, 1])))
        pitch = float(np.degrees(np.arctan2(-rmat[2, 0], sy)))
        yaw   = 0.0

    return (yaw, pitch, roll)


def build_lstm_feature(
    face_lm,
    image_size: tuple[int, int],
) -> np.ndarray | None:
    """Build the fixed LSTM input vector from a FaceMesh landmark list.

    Feature order (source of truth for all callers):
        face_xyz_flatten (1404) → ear (1) → mar (1) → yaw (1) → pitch (1) → roll (1)
        total = config.LSTM_FEATURE_DIM = 1409

    Args:
        face_lm: MediaPipe NormalizedLandmarkList, or None.
        image_size: (width, height) of the source frame in pixels.

    Returns:
        float32 ndarray of shape (LSTM_FEATURE_DIM,) = (1409,),
        or None if face_lm is None.
    """
    if face_lm is None:
        return None

    lm = face_lm.landmark[: config.FACEMESH_LANDMARK_COUNT]
    face_vec = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32).flatten()

    ear = np.float32(compute_ear(face_lm))
    mar = np.float32(compute_mar(face_lm))
    yaw, pitch, roll = estimate_head_pose(face_lm, image_size)

    return np.concatenate(
        [face_vec, [ear], [mar], [yaw], [pitch], [roll]]
    ).astype(np.float32)



# ---------------------------------------------------------------------------
# 이상 자세 탐지 모듈용 헬퍼 (autoencoder)
# ---------------------------------------------------------------------------

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def extract_landmarks(results) -> np.ndarray | None:
    if not results.pose_landmarks:
        return None
    landmarks = results.pose_landmarks.landmark
    vector = []
    for lm in landmarks:
        vector.extend([lm.x, lm.y, lm.z])
    return np.array(vector, dtype=np.float32)


def normalize_landmarks(vector: np.ndarray) -> np.ndarray:
    vec = vector.copy().reshape(33, 3)
    left_shoulder  = vec[11]
    right_shoulder = vec[12]
    center = (left_shoulder + right_shoulder) / 2.0
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder) + 1e-8
    vec = (vec - center) / shoulder_width
    return vec.flatten().astype(np.float32)


def draw_landmarks(frame, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )
    return frame
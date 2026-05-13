from __future__ import annotations

"""공통 전처리 유틸리티.

로컬 mp4 파일을 읽어 MediaPipe feature를 추출하는 모델 공통 helper만 둔다.
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np

import config
from src.utils.mediapipe_utils import (
    build_lstm_feature,
    extract_pose_landmarks,
    normalize_landmarks,
    process_face,
)


logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

POSTURE_LABEL_DIRS = {
    "0_normal": 0,
    "normal": 0,
    "1_turtle_neck": 1,
    "turtle_neck": 1,
    "2_severe_turtle_neck": 2,
    "severe_turtle_neck": 2,
}

FOCUS_LABEL_DIRS = {
    "0_normal": 0,
    "normal": 0,
    "focused": 0,
    "1_drowsy": 1,
    "drowsy": 1,
    "2_distracted": 2,
    "distracted": 2,
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def iter_labeled_videos(raw_root: Path, label_dirs: dict[str, int]) -> list[tuple[Path, int]]:
    videos: list[tuple[Path, int]] = []
    if not raw_root.exists():
        return videos

    for label_dir, label in label_dirs.items():
        folder = raw_root / label_dir
        if not folder.exists():
            continue
        for ext in VIDEO_EXTENSIONS:
            videos.extend((path, label) for path in sorted(folder.glob(f"*{ext}")))
            videos.extend((path, label) for path in sorted(folder.glob(f"*{ext.upper()}")))

    return sorted(videos, key=lambda item: str(item[0]))


def video_frame_step(cap: cv2.VideoCapture, target_fps: int) -> int:
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0 or np.isnan(src_fps):
        src_fps = 30.0
    return max(1, int(round(src_fps / max(1, target_fps))))


def extract_pose_video_features(
    video_path: Path,
    *,
    normalize: bool,
    target_fps: int,
    max_frames: int | None,
) -> tuple[np.ndarray, dict[str, int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"영상 파일을 열 수 없습니다: {video_path}")

    step = video_frame_step(cap, target_fps)
    features: list[np.ndarray] = []
    total_frames = 0
    sampled_frames = 0
    skipped_no_pose = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            if (total_frames - 1) % step != 0:
                continue

            sampled_frames += 1
            pose_vec, _ = extract_pose_landmarks(frame, mode="dashboard")
            if pose_vec is None:
                skipped_no_pose += 1
                continue

            if normalize:
                pose_vec = normalize_landmarks(pose_vec)
            features.append(pose_vec.astype(np.float32))

            if max_frames is not None and len(features) >= max_frames:
                break
    finally:
        cap.release()

    meta = {
        "total_frames": total_frames,
        "sampled_frames": sampled_frames,
        "valid_frames": len(features),
        "skipped_no_pose": skipped_no_pose,
    }
    if not features:
        return np.empty((0, config.POSE_LANDMARK_DIM), dtype=np.float32), meta
    return np.array(features, dtype=np.float32), meta


def extract_lstm_video_features(
    video_path: Path,
    *,
    target_fps: int,
    max_frames: int | None,
) -> tuple[np.ndarray, dict[str, int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"영상 파일을 열 수 없습니다: {video_path}")

    step = video_frame_step(cap, target_fps)
    features: list[np.ndarray] = []
    total_frames = 0
    sampled_frames = 0
    skipped_no_face = 0
    skipped_feature_error = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            if (total_frames - 1) % step != 0:
                continue

            sampled_frames += 1
            frame_size = (frame.shape[1], frame.shape[0])
            face_lm = process_face(frame, mode="dashboard")
            if face_lm is None:
                skipped_no_face += 1
                continue

            try:
                feature = build_lstm_feature(face_lm, frame_size)
            except Exception as exc:
                skipped_feature_error += 1
                logger.warning("LSTM feature 생성 실패: %s (%s)", video_path, exc)
                continue

            if feature is None:
                skipped_feature_error += 1
                continue
            features.append(feature.astype(np.float32))

            if max_frames is not None and len(features) >= max_frames:
                break
    finally:
        cap.release()

    meta = {
        "total_frames": total_frames,
        "sampled_frames": sampled_frames,
        "valid_frames": len(features),
        "skipped_no_face": skipped_no_face,
        "skipped_feature_error": skipped_feature_error,
    }
    if not features:
        return np.empty((0, config.LSTM_FEATURE_DIM), dtype=np.float32), meta
    return np.array(features, dtype=np.float32), meta


def save_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

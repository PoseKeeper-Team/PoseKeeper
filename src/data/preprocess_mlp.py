from __future__ import annotations

"""MLP 거북목 분류 데이터 전처리."""

import logging

import numpy as np
import pandas as pd

import config
from src.data.preprocess_common import (
    POSTURE_LABEL_DIRS,
    extract_pose_video_features,
    iter_labeled_videos,
    save_json,
)


logger = logging.getLogger(__name__)


def preprocess_mlp(
    target_fps: int = 6,
    max_frames_per_video: int | None = None,
) -> None:
    """MLP 거북목 탐지용 전처리.

    입력: data/raw/posture/*/*.mp4 또는 TurtleNeckMLP/data.csv
    출력: data/processed/mlp/X.npy, y.npy
    """
    raw_root = config.PATHS["raw"] / "posture"
    out_dir = config.PATHS["processed"] / "mlp"
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = iter_labeled_videos(raw_root, POSTURE_LABEL_DIRS)
    if videos:
        X: list[np.ndarray] = []
        y: list[int] = []
        video_meta: list[dict] = []

        for video_path, label in videos:
            try:
                features, meta = extract_pose_video_features(
                    video_path,
                    normalize=False,
                    target_fps=target_fps,
                    max_frames=max_frames_per_video,
                )
            except Exception as exc:
                logger.warning("[MLP] 영상 처리 실패: %s (%s)", video_path, exc)
                continue

            if len(features) == 0:
                logger.warning("[MLP] 유효 Pose 없음: %s", video_path)
                continue

            X.append(features)
            y.extend([label] * len(features))
            video_meta.append({"path": str(video_path), "label": label, **meta})
            logger.info(
                "[MLP] %s | label=%s | valid=%s | skipped_no_pose=%s",
                video_path.name,
                label,
                meta["valid_frames"],
                meta["skipped_no_pose"],
            )

        if not X:
            raise RuntimeError("[MLP] mp4에서 추출된 샘플이 없습니다.")

        X_arr = np.concatenate(X, axis=0).astype(np.float32)
        y_arr = np.array(y, dtype=np.int64)

        if X_arr.shape[1] != config.POSE_LANDMARK_DIM:
            raise ValueError(
                f"[MLP] feature dim 오류: {X_arr.shape[1]} != {config.POSE_LANDMARK_DIM}"
            )

        np.save(out_dir / "X.npy", X_arr)
        np.save(out_dir / "y.npy", y_arr)
        save_json(
            out_dir / "meta.json",
            {
                "source": "local_mp4",
                "target_fps": target_fps,
                "feature_dim": config.POSE_LANDMARK_DIM,
                "num_samples": int(len(X_arr)),
                "class_counts": {
                    "0_normal": int((y_arr == 0).sum()),
                    "1_turtle_neck": int((y_arr == 1).sum()),
                    "2_severe_turtle_neck": int((y_arr == 2).sum()),
                },
                "videos": video_meta,
            },
        )
        logger.info("[MLP] 저장 완료: X=%s y=%s out=%s", X_arr.shape, y_arr.shape, out_dir)
        return

    raw_path = config.PROJECT_ROOT / "TurtleNeckMLP" / "data.csv"
    if not raw_path.exists():
        print(f"[FAIL] Raw data not found at {raw_path}")
        return

    try:
        df = pd.read_csv(raw_path)
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.int64)

        X_flipped = X.copy()
        X_flipped[:, 0::3] = 1.0 - X_flipped[:, 0::3]

        X_final = np.concatenate([X, X_flipped], axis=0)
        y_final = np.concatenate([y, y], axis=0)

        np.save(out_dir / "X.npy", X_final)
        np.save(out_dir / "y.npy", y_final)
        save_json(
            out_dir / "meta.json",
            {
                "source": str(raw_path),
                "feature_dim": config.POSE_LANDMARK_DIM,
                "num_samples": int(len(X_final)),
                "class_counts": {
                    "0_normal": int((y_final == 0).sum()),
                    "1_turtle_neck": int((y_final == 1).sum()),
                    "2_severe_turtle_neck": int((y_final == 2).sum()),
                },
            },
        )
        print(f"[MLP] Preprocessing complete. Samples: {len(X)} -> {len(X_final)}")
        print(f"[MLP] Saved to {out_dir}")
    except Exception as e:
        print(f"[MLP] Preprocessing failed: {e}")

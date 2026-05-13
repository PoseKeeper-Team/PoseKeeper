from __future__ import annotations

"""Autoencoder 이상 자세 탐지 데이터 전처리."""

import argparse
import logging

import numpy as np

import config
from src.data.preprocess_common import (
    POSTURE_LABEL_DIRS,
    extract_pose_video_features,
    iter_labeled_videos,
    save_json,
    setup_logging,
)


logger = logging.getLogger(__name__)


def preprocess_ae(
    target_fps: int = 6,
    max_frames_per_video: int | None = None,
    val_ratio: float = 0.1,
) -> None:
    """Autoencoder 비정형 이상자세 탐지용 전처리.

    입력: data/raw/posture/0_normal/*.mp4 또는 data/raw/normal_poses.npy
    출력: data/processed/ae/normal_poses.npy
    """
    raw_path = config.PATHS["raw"] / "normal_poses.npy"
    out_dir = config.PATHS["processed"] / "ae"
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_out_dir = config.PATHS["processed"] / "autoencoder"
    spec_out_dir.mkdir(parents=True, exist_ok=True)

    raw_root = config.PATHS["raw"] / "posture"
    normal_videos = [
        (path, label)
        for path, label in iter_labeled_videos(raw_root, POSTURE_LABEL_DIRS)
        if label == 0
    ]

    if normal_videos:
        X: list[np.ndarray] = []
        video_meta: list[dict] = []

        for video_path, _ in normal_videos:
            try:
                features, meta = extract_pose_video_features(
                    video_path,
                    normalize=True,
                    target_fps=target_fps,
                    max_frames=max_frames_per_video,
                )
            except Exception as exc:
                logger.warning("[AE] 영상 처리 실패: %s (%s)", video_path, exc)
                continue

            if len(features) == 0:
                logger.warning("[AE] 유효 Pose 없음: %s", video_path)
                continue

            X.append(features)
            video_meta.append({"path": str(video_path), "label": 0, **meta})
            logger.info(
                "[AE] %s | valid=%s | skipped_no_pose=%s",
                video_path.name,
                meta["valid_frames"],
                meta["skipped_no_pose"],
            )

        if not X:
            raise RuntimeError("[AE] mp4에서 추출된 정상 자세 샘플이 없습니다.")

        data = np.concatenate(X, axis=0).astype(np.float32)
        if data.shape[1] != config.POSE_LANDMARK_DIM:
            raise ValueError(
                f"[AE] feature dim 오류: {data.shape[1]} != {config.POSE_LANDMARK_DIM}"
            )

        np.save(out_dir / "normal_poses.npy", data)

        rng = np.random.default_rng(42)
        indices = rng.permutation(len(data))
        n_val = max(1, int(len(data) * val_ratio))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        if len(train_idx) == 0:
            train_idx = val_idx

        np.save(spec_out_dir / "X_train.npy", data[train_idx])
        np.save(spec_out_dir / "X_val.npy", data[val_idx])
        meta = {
            "source": "local_mp4",
            "target_fps": target_fps,
            "feature_dim": config.POSE_LANDMARK_DIM,
            "num_samples": int(len(data)),
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "videos": video_meta,
        }
        save_json(spec_out_dir / "meta.json", meta)
        save_json(out_dir / "meta.json", meta)
        logger.info("[AE] 저장 완료: normal_poses=%s out=%s", data.shape, out_dir)
        return

    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} 가 없습니다. 먼저 collect.py로 데이터를 수집하세요."
        )

    data = np.load(raw_path).astype(np.float32)
    print(f"[로드] {raw_path}  shape: {data.shape}")

    flipped = data.copy()
    flipped[:, 0::3] = 1.0 - flipped[:, 0::3]

    noise = np.random.normal(0, 0.01, data.shape).astype(np.float32)
    noisy = data + noise

    augmented = np.concatenate([data, flipped, noisy], axis=0)
    np.random.shuffle(augmented)

    out_path = out_dir / "normal_poses.npy"
    np.save(out_path, augmented)
    save_json(
        out_dir / "meta.json",
        {
            "source": str(raw_path),
            "feature_dim": config.POSE_LANDMARK_DIM,
            "num_samples": int(len(augmented)),
        },
    )

    print(f"[전처리 완료] 원본 {len(data)}개 -> 증강 후 {len(augmented)}개")
    print(f"[저장] {out_path}")


def main() -> None:
    """Run AE preprocessing directly with `python -m src.data.preprocess_ae`."""
    setup_logging()

    parser = argparse.ArgumentParser(description="Autoencoder 이상자세 데이터 전처리")
    parser.add_argument(
        "--target-fps",
        type=int,
        default=6,
        help="mp4 분석 시 초당 샘플링할 프레임 수",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=None,
        help="디버깅용: 영상별 최대 유효 feature 수 제한",
    )
    parser.add_argument(
        "--ae-val-ratio",
        type=float,
        default=0.1,
        help="X_train/X_val 분할 검증 비율",
    )
    args = parser.parse_args()

    preprocess_ae(
        target_fps=args.target_fps,
        max_frames_per_video=args.max_frames_per_video,
        val_ratio=args.ae_val_ratio,
    )


if __name__ == "__main__":
    main()

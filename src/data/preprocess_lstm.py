from __future__ import annotations

"""LSTM 집중도 분류 데이터 전처리."""

import logging

import numpy as np

import config
from src.data.preprocess_common import (
    FOCUS_LABEL_DIRS,
    extract_lstm_video_features,
    iter_labeled_videos,
    save_json,
)


logger = logging.getLogger(__name__)


def make_windows(features: np.ndarray, seq_len: int, stride: int) -> list[np.ndarray]:
    """긴 frame feature 배열을 LSTM 입력용 고정 길이 시퀀스로 자른다."""
    windows: list[np.ndarray] = []

    if len(features) < seq_len:
        return windows

    for start in range(0, len(features) - seq_len + 1, stride):
        end = start + seq_len
        windows.append(features[start:end])

    return windows


def preprocess_lstm_focus(
    seq_len: int = config.LSTM_SEQUENCE_LENGTH,
    stride: int = 5,
    target_fps: int = 6,
    max_frames_per_video: int | None = None,
) -> None:
    """LSTM 집중도 분석용 전처리.

    입력: data/raw/drowsiness/*/*.mp4 또는 data/raw/drowsiness/*/*.npz
    출력: data/processed/lstm/X.npy, y.npy, meta.json
    """
    raw_root = config.PATHS["raw"] / "drowsiness"
    out_dir = config.PATHS["processed"] / "lstm"
    out_dir.mkdir(parents=True, exist_ok=True)

    X: list[np.ndarray] = []
    y: list[int] = []

    legacy_focus_root = config.PATHS["raw"] / "focus"
    if not raw_root.exists() and not legacy_focus_root.exists():
        raise FileNotFoundError(
            f"{raw_root} 또는 {legacy_focus_root} 폴더가 없습니다. 먼저 raw 영상을 배치하세요."
        )

    mp4_files = iter_labeled_videos(raw_root, FOCUS_LABEL_DIRS)
    mp4_files.extend(iter_labeled_videos(legacy_focus_root, FOCUS_LABEL_DIRS))

    if mp4_files:
        video_meta: list[dict] = []

        for video_path, label in mp4_files:
            try:
                features, meta = extract_lstm_video_features(
                    video_path,
                    target_fps=target_fps,
                    max_frames=max_frames_per_video,
                )
            except Exception as exc:
                logger.warning("[LSTM] 영상 처리 실패: %s (%s)", video_path, exc)
                continue

            if features.ndim != 2 or features.shape[1] != config.LSTM_FEATURE_DIM:
                raise ValueError(
                    f"{video_path} feature dim 오류: {features.shape} != (*, {config.LSTM_FEATURE_DIM})"
                )

            windows = make_windows(
                features=features,
                seq_len=seq_len,
                stride=stride,
            )
            if not windows:
                logger.warning("[LSTM] 너무 짧음: %s, shape=%s", video_path, features.shape)
                continue

            for window in windows:
                X.append(window)
                y.append(label)

            video_meta.append(
                {"path": str(video_path), "label": label, "windows": len(windows), **meta}
            )
            logger.info(
                "[LSTM] %s | label=%s | valid=%s | windows=%s | skipped_no_face=%s",
                video_path.name,
                label,
                meta["valid_frames"],
                len(windows),
                meta["skipped_no_face"],
            )

        if not X:
            raise RuntimeError("[LSTM] mp4에서 생성된 시퀀스가 없습니다.")

        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.int64)

        np.save(out_dir / "X.npy", X_arr)
        np.save(out_dir / "y.npy", y_arr)

        meta = {
            "source": "local_mp4",
            "target_fps": target_fps,
            "seq_len": seq_len,
            "stride": stride,
            "feature_dim": config.LSTM_FEATURE_DIM,
            "num_samples": int(len(X_arr)),
            "class_counts": {
                "0_normal": int((y_arr == 0).sum()),
                "1_drowsy": int((y_arr == 1).sum()),
                "2_distracted": int((y_arr == 2).sum()),
            },
            "videos": video_meta,
        }
        save_json(out_dir / "meta.json", meta)
        logger.info("[LSTM] 저장 완료: X=%s y=%s out=%s", X_arr.shape, y_arr.shape, out_dir)
        return

    npz_files = sorted(raw_root.glob("*/*.npz"))

    if not npz_files:
        raise RuntimeError(
            f"{raw_root} 안에 .mp4 또는 .npz 파일이 없습니다. 먼저 데이터를 배치하세요."
        )

    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=False)

        features = data["features"].astype(np.float32)
        label = int(data["label"])

        data.close()

        if features.ndim != 2:
            raise ValueError(f"{npz_path} features 차원 오류: {features.shape}")

        if features.shape[1] != config.LSTM_FEATURE_DIM:
            raise ValueError(
                f"{npz_path} feature dim 오류: "
                f"{features.shape[1]} != {config.LSTM_FEATURE_DIM}"
            )

        windows = make_windows(
            features=features,
            seq_len=seq_len,
            stride=stride,
        )

        if not windows:
            print(f"[건너뜀] 너무 짧음: {npz_path.name}, shape={features.shape}")
            continue

        for window in windows:
            X.append(window)
            y.append(label)

        print(
            f"[OK] {npz_path.name} | "
            f"label={label} | frames={len(features)} | windows={len(windows)}"
        )

    if not X:
        raise RuntimeError("전처리 결과가 비어 있습니다. 수집 프레임 수를 늘려주세요.")

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int64)

    np.save(out_dir / "X.npy", X_arr)
    np.save(out_dir / "y.npy", y_arr)

    save_json(
        out_dir / "meta.json",
        {
            "source": "npz",
            "seq_len": seq_len,
            "stride": stride,
            "feature_dim": config.LSTM_FEATURE_DIM,
            "num_samples": int(len(X_arr)),
            "class_counts": {
                "0_normal": int((y_arr == 0).sum()),
                "1_drowsy": int((y_arr == 1).sum()),
                "2_distracted": int((y_arr == 2).sum()),
            },
        },
    )

    print("\n[전처리 완료]")
    print(f"X shape: {X_arr.shape}")
    print(f"y shape: {y_arr.shape}")
    print(f"저장 위치: {out_dir}")

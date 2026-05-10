from __future__ import annotations

"""데이터 전처리 — 히스토그램 평활화, 좌우 반전·밝기 증강.

Owner: 김병훈. data/raw/ → data/processed/ 변환.
"""
"""데이터 전처리 — raw 데이터를 processed 데이터셋으로 변환.

모델별 역할:
- MLP: 거북목 탐지
- LSTM: 집중도 분석, 정상/졸음/딴짓
- Autoencoder: 비정형 이상자세 탐지

현재 이 파일에서는 LSTM 집중도 분석용 전처리를 먼저 구현한다.
MLP/Autoencoder 전처리는 구조만 열어두고 TODO로 남긴다.
"""

import argparse
import json
import os
import pandas as pd
import numpy as np

import config


def preprocess_mlp() -> None:
    """MLP 거북목 탐지용 전처리.
    
    입력: TurtleNeckMLP/data.csv (Raw)
    출력: data/processed/mlp/X.npy, y.npy (Processed)
    """
    raw_path = config.PROJECT_ROOT / "TurtleNeckMLP" / "data.csv"
    out_dir = config.PATHS["processed"] / "mlp"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        print(f"[FAIL] Raw data not found at {raw_path}")
        return

    try:
        df = pd.read_csv(raw_path)
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.int64)

        # 좌우 반전 증강 (99차원: x, y, z 반복 중 x만 반전)
        X_flipped = X.copy()
        X_flipped[:, 0::3] = 1.0 - X_flipped[:, 0::3]

        X_final = np.concatenate([X, X_flipped], axis=0)
        y_final = np.concatenate([y, y], axis=0)

        np.save(out_dir / "X.npy", X_final)
        np.save(out_dir / "y.npy", y_final)
        print(f"[MLP] Preprocessing complete. Samples: {len(X)} -> {len(X_final)}")
        print(f"[MLP] Saved to {out_dir}")
    except Exception as e:
        print(f"[MLP] Preprocessing failed: {e}")


def preprocess_ae() -> None:
    """
    Autoencoder 비정형 이상자세 탐지용 전처리.

    입력: data/raw/normal_poses.npy
    출력: data/processed/ae/normal_poses.npy
    """
    raw_path = config.PATHS["raw"] / "normal_poses.npy"
    out_dir  = config.PATHS["processed"] / "ae"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} 가 없습니다. 먼저 collect.py로 데이터를 수집하세요."
        )

    data = np.load(raw_path).astype(np.float32)
    print(f"[로드] {raw_path}  shape: {data.shape}")

    # ── 좌우 반전 증강 ──────────────────────────────────────────
    # x 좌표만 반전 (x, y, z 중 x = 인덱스 0, 3, 6 ...)
    flipped = data.copy()
    flipped[:, 0::3] = 1.0 - flipped[:, 0::3]

    # ── 밝기 조절 증강 (좌표에 작은 노이즈 추가) ────────────────
    noise = np.random.normal(0, 0.01, data.shape).astype(np.float32)
    noisy = data + noise

    # ── 합치기 ──────────────────────────────────────────────────
    augmented = np.concatenate([data, flipped, noisy], axis=0)
    np.random.shuffle(augmented)

    out_path = out_dir / "normal_poses.npy"
    np.save(out_path, augmented)

    print(f"[전처리 완료] 원본 {len(data)}개 → 증강 후 {len(augmented)}개")
    print(f"[저장] {out_path}")


def _make_windows(features: np.ndarray, seq_len: int, stride: int) -> list[np.ndarray]:
    """
    긴 frame feature 배열을 LSTM 입력용 고정 길이 시퀀스로 자른다.

    예:
        features shape = (720, 1409)
        seq_len = 30
        stride = 5

    결과:
        (0~29), (5~34), (10~39) ... 식의 시퀀스 여러 개
    """
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
) -> None:
    """
    LSTM 집중도 분석용 전처리.

    입력:
        data/raw/drowsiness/0_normal/*.npz
        data/raw/drowsiness/1_drowsy/*.npz
        data/raw/drowsiness/2_distracted/*.npz

    출력:
        data/processed/lstm/X.npy
        data/processed/lstm/y.npy
        data/processed/lstm/meta.json
    """
    raw_root = config.PATHS["raw"] / "drowsiness"
    out_dir = config.PATHS["processed"] / "lstm"
    out_dir.mkdir(parents=True, exist_ok=True)

    X: list[np.ndarray] = []
    y: list[int] = []

    if not raw_root.exists():
        raise FileNotFoundError(
            f"{raw_root} 폴더가 없습니다. 먼저 collect.py로 LSTM 데이터를 수집하세요."
        )

    npz_files = sorted(raw_root.glob("*/*.npz"))

    if not npz_files:
        raise RuntimeError(
            f"{raw_root} 안에 .npz 파일이 없습니다. 먼저 데이터를 수집하세요."
        )

    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=False)

        features = data["features"].astype(np.float32)
        label = int(data["label"])

        data.close()

        if features.ndim != 2:
            raise ValueError(
                f"{npz_path} features 차원 오류: {features.shape}"
            )

        if features.shape[1] != config.LSTM_FEATURE_DIM:
            raise ValueError(
                f"{npz_path} feature dim 오류: "
                f"{features.shape[1]} != {config.LSTM_FEATURE_DIM}"
            )

        windows = _make_windows(
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

    meta = {
        "seq_len": seq_len,
        "stride": stride,
        "feature_dim": config.LSTM_FEATURE_DIM,
        "num_samples": int(len(X_arr)),
        "class_counts": {
            "0_normal": int((y_arr == 0).sum()),
            "1_drowsy": int((y_arr == 1).sum()),
            "2_distracted": int((y_arr == 2).sum()),
        },
    }

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n[전처리 완료]")
    print(f"X shape: {X_arr.shape}")
    print(f"y shape: {y_arr.shape}")
    print(f"저장 위치: {out_dir}")
    print(f"class counts: {meta['class_counts']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PoseKeeper 데이터 전처리 도구")

    parser.add_argument(
        "--task",
        choices=["mlp", "lstm", "ae", "all"],
        required=True,
        help="mlp: 거북목, lstm: 집중도, ae: 이상자세, all: 전체 전처리",
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=config.LSTM_SEQUENCE_LENGTH,
        help="LSTM 시퀀스 길이",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="LSTM 시퀀스를 자를 때 이동 간격",
    )

    args = parser.parse_args()

    if args.task == "mlp":
        preprocess_mlp()

    elif args.task == "lstm":
        preprocess_lstm_focus(
            seq_len=args.seq_len,
            stride=args.stride,
        )

    elif args.task == "ae":
        preprocess_ae()

    elif args.task == "all":
        preprocess_mlp()
        preprocess_lstm_focus(
            seq_len=args.seq_len,
            stride=args.stride,
        )
        preprocess_ae()


if __name__ == "__main__":
    main()
from __future__ import annotations

"""PoseKeeper 데이터 전처리 CLI.

모델별 구현은 파일을 분리해 둔다.
- MLP: src.data.preprocess_mlp
- LSTM: src.data.preprocess_lstm
- Autoencoder: src.data.preprocess_ae
"""

import argparse

import config
from src.data.preprocess_ae import preprocess_ae
from src.data.preprocess_common import setup_logging
from src.data.preprocess_lstm import preprocess_lstm_focus
from src.data.preprocess_mlp import preprocess_mlp


def main() -> None:
    setup_logging()

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
        help="AE 명세용 X_train/X_val 분할 비율",
    )

    args = parser.parse_args()

    if args.task == "mlp":
        preprocess_mlp(
            target_fps=args.target_fps,
            max_frames_per_video=args.max_frames_per_video,
        )

    elif args.task == "lstm":
        preprocess_lstm_focus(
            seq_len=args.seq_len,
            stride=args.stride,
            target_fps=args.target_fps,
            max_frames_per_video=args.max_frames_per_video,
        )

    elif args.task == "ae":
        preprocess_ae(
            target_fps=args.target_fps,
            max_frames_per_video=args.max_frames_per_video,
            val_ratio=args.ae_val_ratio,
        )

    elif args.task == "all":
        preprocess_mlp(
            target_fps=args.target_fps,
            max_frames_per_video=args.max_frames_per_video,
        )
        preprocess_lstm_focus(
            seq_len=args.seq_len,
            stride=args.stride,
            target_fps=args.target_fps,
            max_frames_per_video=args.max_frames_per_video,
        )
        preprocess_ae(
            target_fps=args.target_fps,
            max_frames_per_video=args.max_frames_per_video,
            val_ratio=args.ae_val_ratio,
        )


if __name__ == "__main__":
    main()

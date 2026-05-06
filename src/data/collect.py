"""웹캠 데이터 수집 + 라벨링 도구.

Owner: 김병훈. config.TRAIN_CAPTURE_RESOLUTION 사용, data/raw/ 에 저장.
"""

"""
data_collector.py
------------------
웹캠으로 '정상 자세' 랜드마크 데이터를 수집하고 .npy 파일로 저장한다.

사용법:
    python data_collector.py --output data/normal_poses.npy --frames 3000

조작키:
    SPACE : 수집 일시 정지 / 재개
    q     : 수집 종료 및 저장
"""

import argparse
import os
import time

import cv2
import numpy as np

from config import PATHS
from src.utils.mediapipe_utils import extract_landmarks, normalize_landmarks, draw_landmarks, mp_pose


def collect(output_path: str, target_frames: int, show_preview: bool = True):
    """
    Args:
        output_path   : 저장 경로 (.npy)
        target_frames : 목표 수집 프레임 수
        show_preview  : 웹캠 미리보기 표시 여부
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    collected: list[np.ndarray] = []
    paused = False

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"\n[데이터 수집 시작] 목표: {target_frames} 프레임")
    print("  SPACE : 일시 정지 / 재개")
    print("  q     : 종료 및 저장\n")

    with mp_pose.Pose(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # 좌우 반전 (거울 모드)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            # ── 랜드마크 추출 및 저장 ──────────────────────────
            if not paused:
                vec = extract_landmarks(results)
                if vec is not None:
                    vec = normalize_landmarks(vec)
                    collected.append(vec)

            # ── 미리보기 ────────────────────────────────────────
            if show_preview:
                draw_landmarks(frame, results)

                status = "⏸ 일시정지" if paused else "● 수집 중"
                n = len(collected)
                pct = min(n / target_frames * 100, 100)

                # 상태 텍스트
                cv2.putText(frame, f"{status}  {n}/{target_frames} ({pct:.1f}%)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0) if not paused else (0, 165, 255), 2)

                # 진행 바
                bar_w = int(620 * pct / 100)
                cv2.rectangle(frame, (10, 450), (630, 470), (50, 50, 50), -1)
                cv2.rectangle(frame, (10, 450), (10 + bar_w, 470), (0, 200, 100), -1)

                cv2.imshow("정상 자세 데이터 수집 (q: 저장 종료)", frame)

            # ── 키 입력 ─────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused
                print("  [일시정지]" if paused else "  [재개]")

            # ── 목표 달성 ────────────────────────────────────────
            if len(collected) >= target_frames:
                print(f"\n[목표 달성] {target_frames} 프레임 수집 완료!")
                break

    cap.release()
    cv2.destroyAllWindows()

    # ── 저장 ────────────────────────────────────────────────────
    if collected:
        data = np.array(collected, dtype=np.float32)
        np.save(output_path, data)
        print(f"[저장 완료] {output_path}  shape: {data.shape}")
    else:
        print("[경고] 수집된 데이터가 없습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="정상 자세 데이터 수집")
    parser.add_argument("--output",  default=str(PATHS["raw"] / "normal_poses.npy"), help="저장 경로")
    parser.add_argument("--frames",  type=int, default=3000,          help="목표 수집 프레임 수")
    parser.add_argument("--no-preview", action="store_true",          help="미리보기 창 숨김")
    args = parser.parse_args()

    collect(
        output_path=args.output,
        target_frames=args.frames,
        show_preview=not args.no_preview,
    )

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
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

import config
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

# ---------------------------------------------------------------------
# LSTM 집중도 분석용 FaceMesh 데이터 수집
# label:
#   0 = normal
#   1 = drowsy
#   2 = distracted
#
# feature:
#   468개 FaceMesh landmark x,y,z = 1404
#   EAR = 1
#   MAR = 1
#   yaw, pitch, roll = 3
#   total = 1409
# ---------------------------------------------------------------------

FOCUS_LABEL_NAMES = {
    0: "normal",
    1: "drowsy",
    2: "distracted",
}

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]


def _point2d(landmarks, idx: int, image_w: int, image_h: int) -> np.ndarray:
    lm = landmarks[idx]
    return np.array([lm.x * image_w, lm.y * image_h], dtype=np.float32)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _calculate_ear(landmarks, eye_indices: list[int], image_w: int, image_h: int) -> float:
    """
    EAR(Eye Aspect Ratio)
    눈이 감기면 세로 길이가 줄어들어서 EAR 값이 작아진다.
    """
    p1 = _point2d(landmarks, eye_indices[0], image_w, image_h)
    p2 = _point2d(landmarks, eye_indices[1], image_w, image_h)
    p3 = _point2d(landmarks, eye_indices[2], image_w, image_h)
    p4 = _point2d(landmarks, eye_indices[3], image_w, image_h)
    p5 = _point2d(landmarks, eye_indices[4], image_w, image_h)
    p6 = _point2d(landmarks, eye_indices[5], image_w, image_h)

    vertical_1 = _dist(p2, p6)
    vertical_2 = _dist(p3, p5)
    horizontal = _dist(p1, p4)

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def _calculate_mar(landmarks, image_w: int, image_h: int) -> float:
    """
    MAR(Mouth Aspect Ratio)
    입이 벌어지면 MAR 값이 커진다.
    하품/졸음 보조 피처로 사용한다.
    """
    left = _point2d(landmarks, 61, image_w, image_h)
    right = _point2d(landmarks, 291, image_w, image_h)

    upper = _point2d(landmarks, 13, image_w, image_h)
    lower = _point2d(landmarks, 14, image_w, image_h)

    upper_left = _point2d(landmarks, 81, image_w, image_h)
    lower_left = _point2d(landmarks, 178, image_w, image_h)

    upper_right = _point2d(landmarks, 311, image_w, image_h)
    lower_right = _point2d(landmarks, 308, image_w, image_h)

    horizontal = _dist(left, right)

    if horizontal == 0:
        return 0.0

    vertical_1 = _dist(upper, lower)
    vertical_2 = _dist(upper_left, lower_left)
    vertical_3 = _dist(upper_right, lower_right)

    return (vertical_1 + vertical_2 + vertical_3) / (3.0 * horizontal)


def _estimate_head_pose(landmarks, image_w: int, image_h: int) -> tuple[float, float, float]:
    """
    Head Pose Estimation.
    반환:
        yaw   : 좌우 고개 방향
        pitch : 위아래 고개 방향
        roll  : 얼굴 기울어짐
    """
    image_points = np.array([
        [landmarks[1].x * image_w, landmarks[1].y * image_h],      # nose tip
        [landmarks[152].x * image_w, landmarks[152].y * image_h],  # chin
        [landmarks[33].x * image_w, landmarks[33].y * image_h],    # left eye outer
        [landmarks[263].x * image_w, landmarks[263].y * image_h],  # right eye outer
        [landmarks[61].x * image_w, landmarks[61].y * image_h],    # mouth left
        [landmarks[291].x * image_w, landmarks[291].y * image_h],  # mouth right
    ], dtype=np.float64)

    model_points = np.array([
        [0.0, 0.0, 0.0],
        [0.0, -63.6, -12.5],
        [-43.3, 32.7, -26.0],
        [43.3, 32.7, -26.0],
        [-28.9, -28.9, -24.1],
        [28.9, -28.9, -24.1],
    ], dtype=np.float64)

    focal_length = image_w
    center = (image_w / 2, image_h / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vec, _ = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return 0.0, 0.0, 0.0

    rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

    pitch = float(angles[0])
    yaw = float(angles[1])
    roll = float(angles[2])

    return yaw, pitch, roll


def _build_focus_lstm_feature(landmarks, image_w: int, image_h: int) -> np.ndarray:
    """
    FaceMesh landmarks + EAR + MAR + HeadPose를 하나의 벡터로 만든다.

    최종 shape:
        config.LSTM_FEATURE_DIM == 1409
    """
    if len(landmarks) < config.FACEMESH_LANDMARK_COUNT:
        raise ValueError(f"FaceMesh landmark 개수 부족: {len(landmarks)}")

    # refine_landmarks=True일 때 478개가 나올 수 있으므로 앞 468개만 사용
    landmarks_468 = landmarks[:config.FACEMESH_LANDMARK_COUNT]

    face_xyz = []

    for lm in landmarks_468:
        face_xyz.extend([lm.x, lm.y, lm.z])

    left_ear = _calculate_ear(landmarks_468, LEFT_EYE_IDX, image_w, image_h)
    right_ear = _calculate_ear(landmarks_468, RIGHT_EYE_IDX, image_w, image_h)
    ear = (left_ear + right_ear) / 2.0

    mar = _calculate_mar(landmarks_468, image_w, image_h)

    yaw, pitch, roll = _estimate_head_pose(landmarks_468, image_w, image_h)

    feature = np.array(
        face_xyz + [ear, mar, yaw, pitch, roll],
        dtype=np.float32,
    )

    if feature.shape[0] != config.LSTM_FEATURE_DIM:
        raise ValueError(
            f"LSTM feature dim 오류: {feature.shape[0]} != {config.LSTM_FEATURE_DIM}"
        )

    return feature


def collect_lstm_focus(
    label: int,
    target_frames: int,
    camera: int = 0,
    target_fps: int = 6,
    show_preview: bool = True,
):
    """
    LSTM 집중도 분석용 웹캠 데이터 수집.

    저장 위치:
        data/raw/drowsiness/0_normal/*.npz
        data/raw/drowsiness/1_drowsy/*.npz
        data/raw/drowsiness/2_distracted/*.npz
    """
    if label not in FOCUS_LABEL_NAMES:
        raise ValueError("label은 0(normal), 1(drowsy), 2(distracted) 중 하나여야 합니다.")

    label_name = FOCUS_LABEL_NAMES[label]

    save_dir = PATHS["raw"] / "drowsiness" / f"{label}_{label_name}"
    save_dir.mkdir(parents=True, exist_ok=True)

    collected: list[np.ndarray] = []
    paused = False

    width, height = config.TRAIN_CAPTURE_RESOLUTION

    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다. 외장캠이면 --camera 1로 다시 시도하세요.")

    interval = 1.0 / target_fps
    last_saved_time = 0.0

    print("\n[LSTM 집중도 데이터 수집 시작]")
    print(f"label = {label} ({label_name})")
    print(f"목표 프레임 수 = {target_frames}")
    print(f"target_fps = {target_fps}")
    print(f"저장 위치 = {save_dir}")
    print("SPACE : 일시정지 / 재개")
    print("q     : 종료 및 저장\n")

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image_h, image_w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            feature = None

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = face_landmarks.landmark

                try:
                    feature = _build_focus_lstm_feature(landmarks, image_w, image_h)
                except Exception as e:
                    print(f"[경고] feature 생성 실패: {e}")
                    feature = None

                now = time.time()

                if feature is not None and not paused and now - last_saved_time >= interval:
                    collected.append(feature)
                    last_saved_time = now

                if show_preview:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
                    )

            if show_preview:
                n = len(collected)
                pct = min(n / target_frames * 100, 100)

                status = "PAUSE" if paused else "REC"
                color = (0, 165, 255) if paused else (0, 255, 0)

                cv2.putText(
                    frame,
                    f"{status} | LSTM {label}:{label_name} | {n}/{target_frames} ({pct:.1f}%)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                )

                if feature is not None:
                    base = config.FACEMESH_LANDMARK_COUNT * 3
                    ear = feature[base]
                    mar = feature[base + 1]
                    yaw = feature[base + 2]
                    pitch = feature[base + 3]
                    roll = feature[base + 4]

                    cv2.putText(
                        frame,
                        f"EAR={ear:.3f} MAR={mar:.3f} yaw={yaw:.1f} pitch={pitch:.1f} roll={roll:.1f}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                    )
                else:
                    cv2.putText(
                        frame,
                        "No face detected",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2,
                    )

                bar_w = int(620 * pct / 100)
                cv2.rectangle(frame, (10, 450), (630, 470), (50, 50, 50), -1)
                cv2.rectangle(frame, (10, 450), (10 + bar_w, 470), (0, 200, 100), -1)

                cv2.imshow("LSTM Focus Data Collector", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" "):
                paused = not paused
                print("[일시정지]" if paused else "[재개]")

            if len(collected) >= target_frames:
                print(f"\n[목표 달성] {target_frames} 프레임 수집 완료")
                break

    cap.release()
    cv2.destroyAllWindows()

    if collected:
        data = np.array(collected, dtype=np.float32)

        filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_label{label}.npz"
        save_path = save_dir / filename

        np.savez_compressed(
            save_path,
            features=data,
            label=np.array(label, dtype=np.int64),
            label_name=np.array(label_name),
            target_fps=np.array(target_fps, dtype=np.int64),
        )

        print(f"[저장 완료] {save_path}")
        print(f"shape: {data.shape}")
    else:
        print("[경고] 수집된 데이터가 없습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PoseKeeper 데이터 수집 도구")

    parser.add_argument(
        "--task",
        choices=["pose", "lstm"],
        default="pose",
        help="pose: MLP/Autoencoder용 자세 데이터 수집, lstm: 집중도 분석용 FaceMesh 데이터 수집",
    )
    
    #pose 수집용 옵션
    parser.add_argument("--output",  default=str(PATHS["raw"] / "normal_poses.npy"), help="저장 경로")
    parser.add_argument("--frames",  type=int, default=3000, help="목표 수집 프레임 수")
    parser.add_argument("--no-preview", action="store_true", help="미리보기 창 숨김")
    
    # LSTM 수집용 옵션
    parser.add_argument("--label", type=int, choices=[0, 1, 2], default=None, help="lstm task label: 0 normal, 1 drowsy, 2 distracted",)
    parser.add_argument("--camera", type=int, default=0, help="웹캠 번호. 기본 0, 외장캠이면 1",)
    parser.add_argument("--target-fps", type=int, default=6, help="lstm task에서 초당 저장할 feature 수",)
    
    args = parser.parse_args()

    if args.task == "pose":
        collect(
            output_path=args.output,
            target_frames=args.frames,
            show_preview=not args.no_preview,
        )
    elif args.task == "lstm":
        if args.label is None:
            raise ValueError("--task lstm 사용 시 --label 0/1/2 를 반드시 지정해야 합니다.")
        
        collect_lstm_focus(
            label=args.label,
            target_frames=args.frames,
            camera=args.camera,
            target_fps=args.target_fps,
            show_preview=not args.no_preview,
        )
    

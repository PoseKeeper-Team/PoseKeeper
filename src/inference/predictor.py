"""실시간 추론 통합 모듈.

3개 모델(MLP/LSTM/AE)을 lazy-load하고 한 번의 MediaPipe 처리 결과를
공유해 대시보드·알림·DB가 공통으로 쓸 수 있는 결과 dict를 반환한다.
weight 파일이 없으면 해당 모델만 비활성화하고 앱은 계속 동작한다.
"""
from __future__ import annotations

import logging
import time
from collections import deque

import numpy as np
import torch

import config
from src.utils.mediapipe_utils import (
    extract_pose_landmarks,
    process_face,
    compute_ear,
    compute_mar,
    estimate_head_pose,
    build_lstm_feature,
    normalize_landmarks,
    draw_landmarks,
)

logger = logging.getLogger(__name__)

_FOCUS_LABELS   = {0: "focused", 1: "drowsy", 2: "distracted"}
_PREDICTOR: "PosePredictor | None" = None


class PosePredictor:
    """웹캠 프레임 1장을 받아 MLP/LSTM/AE를 통합 추론하고 결과 dict를 반환한다."""

    def __init__(self) -> None:
        self._mlp_model: object | None = None
        self._lstm_model: object | None = None
        self._ae_model: object | None = None
        self._ae_threshold: float | None = None

        self._loaded = False
        self._lstm_buffer: deque[np.ndarray] = deque(maxlen=config.LSTM_SEQUENCE_LENGTH)
        self._event_timers: dict[str, float] = {}
        self._calibration_pose: np.ndarray | None = None  # 캘리브레이션 기준 포즈
        self._last_result: dict | None = None
        self._last_log_time = 0.0
        self._last_warn_time: dict[str, float] = {}
        
        self._no_face_count = 0
        self._no_face_threshold = 6  # 얼굴이 6프레임 연속 안 잡히면 산만함 후보

    def reset(self) -> None:
        """세션 재시작 시 LSTM 버퍼·이벤트 타이머 초기화."""
        self._lstm_buffer.clear()
        self._event_timers.clear()
        self._last_result = None
        self._no_face_count = 0
        logger.info("PosePredictor reset")

    def calibrate(self, frame: np.ndarray) -> bool:
        """현재 프레임의 포즈를 '정상 자세'의 기준으로 저장합니다."""
        pose_vec, _ = extract_pose_landmarks(frame, mode="dashboard")
        if pose_vec is not None:
            self._calibration_pose = pose_vec
            logger.info("Calibration successful")
            return True
        return False

    def _ensure_loaded(self) -> None:
        """첫 호출 시 1회만 모델을 로드한다. 실패해도 앱을 중단하지 않는다."""
        if self._loaded:
            return
        self._loaded = True

        # MLP (거북목)
        try:
            from src.models.mlp import PoseMLP  # noqa: PLC0415
            path = config.WEIGHT_FILES["mlp"]
            if path.exists():
                self._mlp_model = PoseMLP(input_dim=config.POSE_LANDMARK_DIM, num_classes=3)
                
                # 학습 시 딕셔너리 형태로 저장하므로, 해당 키를 찾아 로드함
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self._mlp_model.load_state_dict(checkpoint["model_state_dict"]) # type: ignore[union-attr]
                    logger.info("MLP dictionary format loaded. Best Acc: %.2f%%", checkpoint.get("best_val_accuracy", 0))
                else:
                    self._mlp_model.load_state_dict(checkpoint) # type: ignore[union-attr]
                
                self._mlp_model.to(config.DEVICE)  # type: ignore[union-attr]
                self._mlp_model.eval()  # type: ignore[union-attr]
                logger.info("MLP loaded: %s", path)
            else:
                logger.warning("MLP weight not found: %s — posture detection disabled", path)
        except Exception as e:
            logger.warning("MLP load failed: %s", e)

        # LSTM (집중도)
        try:
            from src.models.lstm import PoseLSTM  # noqa: PLC0415
            path = config.WEIGHT_FILES["lstm"]
            if path.exists():
                self._lstm_model = PoseLSTM(input_dim=config.LSTM_FEATURE_DIM, num_classes=3)
                self._lstm_model.load_state_dict(  # type: ignore[union-attr]
                    torch.load(path, map_location=config.DEVICE, weights_only=True)
                )
                self._lstm_model.to(config.DEVICE)  # type: ignore[union-attr]
                self._lstm_model.eval()  # type: ignore[union-attr]
                logger.info("LSTM loaded: %s", path)
            else:
                logger.warning("LSTM weight not found: %s — focus detection disabled", path)
        except Exception as e:
            logger.warning("LSTM load failed: %s", e)

        # AE (이상 자세)
        try:
            from src.models.autoencoder import PoseAutoencoder  # noqa: PLC0415
            ae_path  = config.WEIGHT_FILES["autoencoder"]
            thr_path = config.WEIGHT_FILES["autoencoder_threshold"]
            if ae_path.exists() and thr_path.exists():
                self._ae_model = PoseAutoencoder(input_dim=99, latent_dim=16)
                self._ae_model.load_state_dict(  # type: ignore[union-attr]
                    torch.load(ae_path, map_location=config.DEVICE, weights_only=True)
                )
                self._ae_model.to(config.DEVICE)  # type: ignore[union-attr]
                self._ae_model.eval()  # type: ignore[union-attr]
                self._ae_threshold = float(np.load(thr_path)[0])
                logger.info("AE loaded: %s (threshold=%.6f)", ae_path, self._ae_threshold)
            else:
                logger.warning("AE weight not found — anomaly detection disabled")
        except Exception as e:
            logger.warning("AE load failed: %s", e)

    # ------------------------------------------------------------------
    # 결과 헬퍼
    # ------------------------------------------------------------------

    def _build_empty_result(self) -> dict:
        return {
            "timestamp":          time.time(),
            "is_skip":            False,
            "posture_class":      None,
            "posture_label":      None,
            "posture_confidence": None,
            "focus_class":        None,
            "focus_label":        None,
            "focus_confidence":   None,
            "anomaly_score":      None,
            "is_anomaly":         False,
            "pose_score":         100.0,
            "events":             [],
            "event_severity":     {},
        }

    # ------------------------------------------------------------------
    # 이벤트·severity
    # ------------------------------------------------------------------

    def _get_severity(
        self,
        event: str,
        elapsed: float,
        posture_label: str | None = None,
    ) -> str:
        if event == "turtle_neck":
            T = config.TURTLE_NECK_THRESHOLD_SEC
            if posture_label == "severe_turtle_neck" or elapsed >= 3 * T:
                return "high"
            if elapsed >= 2 * T:
                return "medium"
            return "low"

        if event == "drowsy":
            T = config.DROWSINESS_THRESHOLD_SEC
            if elapsed >= 3 * T:
                return "high"
            if elapsed >= 2 * T:
                return "medium"
            return "low"

        if event == "distracted":
            T = config.DISTRACTION_THRESHOLD_SEC
            if elapsed >= 3 * T:
                return "high"
            if elapsed >= 2 * T:
                return "medium"
            return "low"

        if event == "anomaly_posture":
            T = config.TURTLE_NECK_THRESHOLD_SEC
            if elapsed >= 2 * T:
                return "high"
            if elapsed >= T:
                return "medium"
            return "low"

        return "low"

    def _update_event_timers(
        self,
        posture_class: int | None,
        focus_class: int | None,
        is_anomaly: bool,
        ear: float | None,
        mar: float | None,
        yaw: float | None,
        pitch: float | None,
    ) -> None:
        now = time.monotonic()

        # turtle_neck
        if posture_class in {1, 2}:
            self._event_timers.setdefault("turtle_neck", now)
        else:
            self._event_timers.pop("turtle_neck", None)

        # drowsy: LSTM 또는 EAR/MAR 기준
        drowsy_cond = (
            focus_class == 1
            or (ear is not None and ear < config.EAR_THRESHOLD)
            or (mar is not None and mar > config.MAR_THRESHOLD)
        )
        if drowsy_cond:
            self._event_timers.setdefault("drowsy", now)
        else:
            self._event_timers.pop("drowsy", None)

        # distracted: LSTM 또는 HeadPose 기준
        distracted_cond = (
            focus_class == 2
            or (yaw is not None and abs(yaw) > config.HEADPOSE_YAW_THRESHOLD_DEG)
            or (pitch is not None and abs(pitch) > config.HEADPOSE_PITCH_THRESHOLD_DEG)
        )
        if distracted_cond:
            self._event_timers.setdefault("distracted", now)
        else:
            self._event_timers.pop("distracted", None)

        # anomaly_posture
        if is_anomaly:
            self._event_timers.setdefault("anomaly_posture", now)
        else:
            self._event_timers.pop("anomaly_posture", None)

    def _event_threshold(self, event: str) -> float:
        if event == "turtle_neck":
            return config.TURTLE_NECK_THRESHOLD_SEC
        if event == "drowsy":
            return config.DROWSINESS_THRESHOLD_SEC
        if event == "distracted":
            return config.DISTRACTION_THRESHOLD_SEC
        if event == "anomaly_posture":
            return config.TURTLE_NECK_THRESHOLD_SEC
        return 0.0

    def _calc_events_and_severity(
        self,
        posture_label: str | None,
    ) -> tuple[list[str], dict[str, str]]:
        now = time.monotonic()
        events: list[str] = []
        severity: dict[str, str] = {}
        for event, start in self._event_timers.items():
            elapsed = now - start

            # 조건이 잠깐 켜졌다고 바로 이벤트로 띄우지 않고, 설정된 시간 이상 지속됐을 때만 이벤트로 표시한다.
            if elapsed < self._event_threshold(event):
                continue

            events.append(event)
            severity[event] = self._get_severity(event, elapsed, posture_label)
        
        return events, severity

    @staticmethod
    def _calc_pose_score(posture_class: int | None, events: list[str]) -> float:
        score = 100.0
        if posture_class == 2:
            score -= 35
        elif posture_class == 1:
            score -= 20
        if "drowsy" in events:
            score -= 30
        if "distracted" in events:
            score -= 25
        if "anomaly_posture" in events:
            score -= 15
        return max(0.0, min(100.0, score))

    # ------------------------------------------------------------------
    # 메인 추론
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray | None, mode: str = "bg") -> dict:
        """프레임 1장을 받아 3개 모델을 통합 추론하고 결과 dict를 반환한다.

        Args:
            frame: OpenCV BGR ndarray. None이면 빈 결과를 반환한다.
            mode:  "bg" 또는 "dashboard".

        Returns:
            명세서 §4 형식의 결과 dict.
            dashboard 모드이고 Pose가 검출된 경우 "overlay_frame" 키가 추가된다.
        """
        self._ensure_loaded()

        if frame is None:
            return self._build_empty_result()

        frame_size = (frame.shape[1], frame.shape[0])  # (width, height)

        # ── Pose ──────────────────────────────────────────────────────
        pose_vec, pose_results = extract_pose_landmarks(frame, mode)

        # ── Face ──────────────────────────────────────────────────────
        face_lm = process_face(frame, mode)
        no_face_distracted = False

        if face_lm is not None:
            self._no_face_count = 0

            ear: float | None = compute_ear(face_lm)
            mar: float | None = compute_mar(face_lm)
            yaw, pitch, roll = estimate_head_pose(face_lm, frame_size)
            lstm_vec = build_lstm_feature(face_lm, frame_size)
        else:
            self._no_face_count += 1

            # 얼굴이 사라진 상태에서는 이전 LSTM 시퀀스를 계속 쓰면 오판 가능성이 있으므로 비운다.
            self._lstm_buffer.clear()

            ear = mar = yaw = pitch = roll = None
            lstm_vec = None

            if self._no_face_count >= self._no_face_threshold:
                no_face_distracted = True

        # ── MLP (거북목) ───────────────────────────────────────────────
        posture_class: int | None = None
        posture_confidence: float | None = None
        if pose_vec is not None and self._mlp_model is not None:
            try:
                input_vec = pose_vec
                if self._calibration_pose is not None:
                    # 기준 자세(정상)와의 차이를 보정하여 입력값 조정
                    # 99차원 벡터를 (33, 3)으로 변형하여 어깨 등 주요 포인트 중심 이동 보정
                    curr_pose = pose_vec.reshape(-1, 3)
                    base_pose = self._calibration_pose.reshape(-1, 3)
                    
                    # 전체적인 좌표 이동 보정 (기준 자세와의 평균 오프셋 계산)
                    offset = np.mean(base_pose - curr_pose, axis=0)
                    input_vec = (curr_pose + offset).flatten()

                device = next(self._mlp_model.parameters()).device
                x = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = self._mlp_model(x)  # type: ignore[operator]
                probs = torch.softmax(logits, dim=1)[0]
                posture_class = int(probs.argmax().item())
                posture_confidence = float(probs[posture_class].item())
            except Exception as e:
                if time.time() - self._last_warn_time.get("mlp_inf", 0) > 5.0:
                    logger.warning("MLP inference failed: %s", e)
                    self._last_warn_time["mlp_inf"] = time.time()

        # 모델에 정의된 LABELS 참조
        posture_label = None
        if posture_class is not None:
            from src.models.mlp import PoseMLP
            posture_label = PoseMLP.LABELS.get(posture_class)

        # ── LSTM (집중도) ──────────────────────────────────────────────
        focus_class: int | None = None
        focus_confidence: float | None = None

        # 얼굴이 일정 프레임 이상 감지되지 않으면 LSTM 대신 산만함으로 처리
        if no_face_distracted:
            focus_class = 2
            focus_confidence = 1.0

        elif lstm_vec is not None:
            self._lstm_buffer.append(lstm_vec)
            if self._lstm_model is not None and len(self._lstm_buffer) >= config.LSTM_SEQUENCE_LENGTH:
                try:
                    seq = np.stack(list(self._lstm_buffer), axis=0)       # (seq_len, feat)
                    device = next(self._lstm_model.parameters()).device
                    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq, feat)
                    with torch.no_grad():
                        logits = self._lstm_model(x)  # type: ignore[operator]
                    probs = torch.softmax(logits, dim=1)[0]
                    focus_class = int(probs.argmax().item())
                    focus_confidence = float(probs[focus_class].item())
                except Exception as e:
                    logger.warning("LSTM inference failed: %s", e)

        focus_label = _FOCUS_LABELS.get(focus_class) if focus_class is not None else None

        # ── AE (이상 자세) ─────────────────────────────────────────────
        anomaly_score: float | None = None
        is_anomaly = False
        if pose_vec is not None and self._ae_model is not None and self._ae_threshold is not None:
            try:
                normalized = normalize_landmarks(pose_vec)
                device = next(self._ae_model.parameters()).device
                x = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    anomaly_score = float(
                        self._ae_model.reconstruction_error(x).item()  # type: ignore[union-attr]
                    )
                is_anomaly = anomaly_score > self._ae_threshold
            except Exception as e:
                logger.warning("AE inference failed: %s", e)

        # ── 이벤트 통합 ────────────────────────────────────────────────
        self._update_event_timers(posture_class, focus_class, is_anomaly, ear, mar, yaw, pitch)
        events, event_severity = self._calc_events_and_severity(posture_label)
        pose_score = self._calc_pose_score(posture_class, events)

        result: dict = {
            "timestamp":          time.time(),
            "is_skip":            False,
            "posture_class":      posture_class,
            "posture_label":      posture_label,
            "posture_confidence": posture_confidence,
            "focus_class":        focus_class,
            "focus_label":        focus_label,
            "focus_confidence":   focus_confidence,
            "anomaly_score":      anomaly_score,
            "is_anomaly":         is_anomaly,
            "pose_score":         pose_score,
            "events":             events,
            "event_severity":     event_severity,
        }

        # dashboard 모드에서만 랜드마크 오버레이 추가 (호출부 재활용, 추가 MP 호출 없음)
        if mode == "dashboard" and pose_results is not None:
            result["overlay_frame"] = draw_landmarks(frame.copy(), pose_results, face_lm=face_lm)

        self._last_result = result
        
        now = time.time()
        if now - self._last_log_time >= 1.0:
            logger.info(
                "[%s] 자세: %-15s | 집중도: %-10s | 이상여부: %-5s | 점수: %5.1f | 이벤트: %s",
                mode.upper(), 
                (posture_label or "N/A").upper(), 
                (focus_label or "N/A").upper(), 
                "YES" if is_anomaly else "NO", 
                pose_score, 
                events if events else "None"
            )
            self._last_log_time = now
        return result


def get_predictor() -> PosePredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = PosePredictor()
    return _PREDICTOR


def predict(frame: np.ndarray | None, mode: str = "bg") -> dict:
    """Module-level realtime inference entrypoint."""
    return get_predictor().process(frame, mode)


def reset_predictor() -> None:
    global _PREDICTOR
    if _PREDICTOR is not None:
        _PREDICTOR.reset()

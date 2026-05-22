"""pystray 시스템 트레이 — main.py 의 진입점.

메뉴: [상태표시] [대시보드 열기] [일시정지] [설정] [종료]

스레드 모델:
  main thread   → tk.Tk (숨김 루트) mainloop + cmd_queue 폴링
  pystray thread → icon.run_detached() (메뉴 이벤트)
  infer thread  → webcam read → predictor → AlertManager
"""
from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from datetime import datetime, timezone

import cv2
import numpy as np

import config
from src.app.alert import AlertManager
from src.inference.predictor import predict, get_predictor
from src.utils.db import close_session, flush_score_samples, init_db, open_session

logger = logging.getLogger(__name__)

_SAMPLE_FLUSH_INTERVAL = 60.0  # seconds between score_samples flushes

_dashboard: object | None = None  # Dashboard | None (lazy import 회피)


def _fallback_result() -> dict:
    return {
        "timestamp": time.time(),
        "is_skip": False,
        "posture_class": None,
        "posture_label": None,
        "posture_confidence": None,
        "focus_class": None,
        "focus_label": None,
        "focus_confidence": None,
        "anomaly_score": None,
        "is_anomaly": False,
        "pose_score": 100.0,
        "events": [],
        "event_severity": {},
    }


# ---------------------------------------------------------------------------
# 공유 상태
# ---------------------------------------------------------------------------

@dataclass
class SharedState:
    mode: str = "bg"           # "bg" | "dashboard"
    paused: bool = False
    running: bool = True
    camera_ok: bool = False
    camera_retry: bool = False
    calibrate_request: bool = False
    latest_frame: np.ndarray | None = None
    latest_result: dict | None = None
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def update_frame(self, frame: np.ndarray, result: dict) -> None:
        with self._lock:
            self.latest_frame = frame
            self.latest_result = result

    def read_latest(self) -> tuple[np.ndarray | None, dict | None]:
        with self._lock:
            return self.latest_frame, self.latest_result


# ---------------------------------------------------------------------------
# 추론 루프
# ---------------------------------------------------------------------------

def should_process(mode: str, frame_count: int) -> bool:
    skip = config.BG_FRAME_SKIP if mode == "bg" else config.DASHBOARD_FRAME_SKIP
    return frame_count % skip == 0


def inference_loop(
    state: SharedState,
    cap: cv2.VideoCapture,
    alert_manager: AlertManager,
    conn: sqlite3.Connection,
    session_id: int,
) -> None:
    frame_count = 0
    sample_buf: list[dict] = []
    last_flush = time.time()
    consecutive_fail = 0
    _FAIL_THRESHOLD = 30  # ~3초 (0.1s sleep × 30)

    while state.running:
        if state.camera_retry:
            cap.release()
            cap.open(0)
            state.camera_ok = cap.isOpened()
            state.camera_retry = False
            consecutive_fail = 0
            logger.info("카메라 재시도 — %s", "성공" if state.camera_ok else "실패")

        if not state.camera_ok:
            time.sleep(0.5)
            continue

        if state.paused:
            time.sleep(0.1)
            continue

        ok, frame = cap.read()
        if not ok:
            consecutive_fail += 1
            if consecutive_fail >= _FAIL_THRESHOLD:
                state.camera_ok = False
                logger.warning("웹캠 신호 끊김 감지 — 대시보드에서 재시도 가능")
            time.sleep(0.1)
            continue

        if state.calibrate_request:
            get_predictor().calibrate(frame)
            state.calibrate_request = False
            logger.info("실시간 캘리브레이션 완료")

        consecutive_fail = 0
        frame_count += 1
        if not should_process(state.mode, frame_count):
            continue

        try:
            result = predict(frame, state.mode)
        except Exception:
            logger.exception("predictor 실행 실패")
            result = _fallback_result()

        state.update_frame(frame, result)
        alert_manager.process(result)

        score = result.get("pose_score")
        if score is not None:
            sample_buf.append({
                "ts": datetime.now(timezone.utc).isoformat(),
                "score": float(score),
                "posture_label": result.get("posture_label"),
                "focus_label": result.get("focus_label"),
            })

        now = time.time()
        if now - last_flush >= _SAMPLE_FLUSH_INTERVAL and sample_buf:
            try:
                flush_score_samples(conn, session_id, sample_buf)
            except Exception:
                logger.warning("score_samples flush 실패")
            sample_buf.clear()
            last_flush = now

    # 루프 종료 후 남은 샘플 최종 flush
    if sample_buf:
        try:
            flush_score_samples(conn, session_id, sample_buf)
        except Exception:
            logger.warning("score_samples 최종 flush 실패")


# ---------------------------------------------------------------------------
# 트레이 메뉴
# ---------------------------------------------------------------------------

def _status_text(state: SharedState) -> str:
    result = state.latest_result
    if result is None:
        return "PoseKeeper 실행 중"
    score = result.get("pose_score", "-")
    events = result.get("events", [])
    status = ", ".join(events) if events else "정상"
    return f"점수: {score} | {status}"


def _load_icon():
    """트레이 아이콘 이미지 생성 (파란 원)."""
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([4, 4, 60, 60], fill=(30, 144, 255, 255))
    return img


def _build_menu(
    state: SharedState,
    cmd_queue: queue.Queue,
    root: tk.Tk,
    conn: sqlite3.Connection,
    session_id: int,
):
    import pystray

    def on_open_dashboard(icon, item):
        cmd_queue.put(lambda: _open_dashboard(state, root, conn, session_id))

    def on_toggle_pause(icon, item):
        state.paused = not state.paused
        logger.info("추론 %s", "일시정지" if state.paused else "재개")

    def on_open_settings(icon, item):
        logger.info("설정 창 (미구현)")

    def on_quit(icon, item):
        state.running = False
        cmd_queue.put(root.quit)
        icon.stop()

    return pystray.Menu(
        pystray.MenuItem(
            lambda item: _status_text(state),
            lambda icon, item: None,
            enabled=False,
        ),
        pystray.MenuItem("대시보드 열기", on_open_dashboard),
        pystray.MenuItem(
            lambda item: "재개" if state.paused else "일시정지",
            on_toggle_pause,
        ),
        pystray.MenuItem("설정", on_open_settings),
        pystray.MenuItem("종료", on_quit),
    )


def _open_dashboard(
    state: SharedState,
    root: tk.Tk,
    conn: sqlite3.Connection,
    session_id: int,
) -> None:
    """메인 스레드(cmd_queue 경유)에서만 호출된다."""
    global _dashboard

    if _dashboard is not None and _dashboard.window is not None:
        try:
            if _dashboard.window.winfo_exists():
                _dashboard.window.lift()
                return
        except tk.TclError:
            pass

    from src.app.dashboard import Dashboard
    _dashboard = Dashboard(root, state, conn, session_id)
    _dashboard.open()


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

def run_tray() -> None:
    try:
        import pystray  # noqa: F401
    except ImportError:
        print("[오류] pystray가 설치되지 않았습니다: pip install pystray")
        return

    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        print("[오류] Pillow가 설치되지 않았습니다: pip install pillow")
        return

    conn = init_db()
    session_id = open_session(conn)

    state = SharedState()
    alert_manager = AlertManager(conn=conn, session_id=session_id)

    cap = cv2.VideoCapture(0)
    state.camera_ok = cap.isOpened()
    if not state.camera_ok:
        logger.warning("웹캠 초기 열기 실패 — 대시보드에서 재시도 가능")

    root = tk.Tk()
    root.withdraw()

    cmd_queue: queue.Queue = queue.Queue()

    def poll_commands():
        try:
            while True:
                cmd = cmd_queue.get_nowait()
                cmd()
        except queue.Empty:
            pass
        if state.running:
            root.after(50, poll_commands)
        else:
            root.quit()

    root.after(50, poll_commands)

    infer_thread = threading.Thread(
        target=inference_loop,
        args=(state, cap, alert_manager, conn, session_id),
        daemon=True,
    )
    infer_thread.start()

    import pystray
    icon = pystray.Icon(
        "PoseKeeper",
        _load_icon(),
        title="PoseKeeper",
        menu=_build_menu(state, cmd_queue, root, conn, session_id),
    )
    icon.run_detached()

    try:
        root.mainloop()
    finally:
        state.running = False
        icon.stop()
        infer_thread.join(timeout=3)
        if infer_thread.is_alive():
            logger.warning("추론 스레드 미종료 — 강제 정리")
        cap.release()
        try:
            close_session(conn, session_id, avg_score=None)
        except Exception:
            logger.warning("세션 종료 DB 저장 실패")
        conn.close()

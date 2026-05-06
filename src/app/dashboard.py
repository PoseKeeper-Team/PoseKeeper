"""Tkinter 대시보드 — 사용자가 트레이에서 열 때만 활성화.

표시 항목: 웹캠 영상 + 랜드마크 오버레이, 거북목/졸음/딴짓 상태,
자세 점수(0~100), 위험 지속 시간, 시간대별 점수 그래프(matplotlib),
세션 요약.
"""
from __future__ import annotations

import logging
import sqlite3
import time
import tkinter as tk
from tkinter import ttk
from datetime import datetime
from typing import TYPE_CHECKING

import cv2
from PIL import Image, ImageTk

if TYPE_CHECKING:
    from src.app.tray import SharedState

logger = logging.getLogger(__name__)

_STATUS_COLORS: dict[str, str] = {
    "turtle_neck":     "#e67e22",
    "drowsy":          "#e74c3c",
    "distracted":      "#f39c12",
    "anomaly_posture": "#9b59b6",
    "normal":          "#2ecc71",
    "paused":          "#95a5a6",
    "no_camera":       "#c0392b",
}

_VIDEO_W, _VIDEO_H = 480, 360
_BG_DARK  = "#1a1a2e"
_BG_MID   = "#16213e"
_BG_PANEL = "#0f0f23"


class Dashboard:
    def __init__(
        self,
        root: tk.Tk,
        state: "SharedState",
        conn: sqlite3.Connection,
        session_id: int,
    ) -> None:
        self._root = root
        self._state = state
        self._conn = conn
        self._session_id = session_id
        self.window: tk.Toplevel | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._event_start: dict[str, float] = {}
        self._matplotlib_ok = False

    # ------------------------------------------------------------------
    # 공개 인터페이스
    # ------------------------------------------------------------------

    def open(self) -> None:
        win = tk.Toplevel(self._root)
        win.title("PoseKeeper 대시보드")
        win.resizable(False, False)
        win.configure(bg=_BG_DARK)
        self.window = win
        self._build_ui(win)
        win.protocol("WM_DELETE_WINDOW", self._on_close)
        self._state.mode = "dashboard"
        logger.info("대시보드 열림")
        self._update_frame()
        self._update_status()
        self._update_graph()
        self._update_session()

    # ------------------------------------------------------------------
    # 내부 유틸
    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        self._state.mode = "bg"
        logger.info("대시보드 닫힘 — bg 모드 복귀")
        if self.window:
            self.window.destroy()
            self.window = None

    def _alive(self) -> bool:
        try:
            return self.window is not None and bool(self.window.winfo_exists())
        except tk.TclError:
            return False

    # ------------------------------------------------------------------
    # UI 구성
    # ------------------------------------------------------------------

    def _build_ui(self, win: tk.Toplevel) -> None:
        # 왼쪽 패널: 영상 + 상태
        left = tk.Frame(win, bg=_BG_DARK)
        left.pack(side=tk.LEFT, fill=tk.BOTH, padx=8, pady=8)

        video_container = tk.Frame(left, width=_VIDEO_W, height=_VIDEO_H, bg=_BG_PANEL)
        video_container.pack_propagate(False)
        video_container.pack()
        self._video_label = tk.Label(
            video_container, bg=_BG_PANEL,
            text="카메라 초기화 중...", fg="white",
        )
        self._video_label.pack(expand=True)

        self._retry_btn = tk.Button(
            video_container,
            text="카메라 재시도",
            command=self._on_retry_camera,
            bg="#e74c3c", fg="white",
            font=("Arial", 10, "bold"),
            padx=8, pady=4,
        )

        self._status_label = tk.Label(
            left, text="정상", bg=_STATUS_COLORS["normal"], fg="white",
            font=("Arial", 12, "bold"), padx=8, pady=4,
        )
        self._status_label.pack(fill=tk.X, pady=(6, 2))

        score_frame = tk.Frame(left, bg=_BG_DARK)
        score_frame.pack(fill=tk.X, pady=2)
        tk.Label(score_frame, text="자세 점수", bg=_BG_DARK, fg="#aaa", font=("Arial", 9)).pack(side=tk.LEFT)
        self._score_val = tk.Label(score_frame, text="—", bg=_BG_DARK, fg="white", font=("Arial", 13, "bold"))
        self._score_val.pack(side=tk.RIGHT)

        self._score_bar = ttk.Progressbar(left, maximum=100, mode="determinate", length=_VIDEO_W)
        self._score_bar.pack(pady=2)

        self._duration_label = tk.Label(left, text="지속: —", bg=_BG_DARK, fg="#aaa", font=("Arial", 9))
        self._duration_label.pack(anchor=tk.W)

        # 오른쪽 패널: 그래프 + 세션 요약
        right = tk.Frame(win, bg=_BG_MID)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(0, 8), pady=8)

        self._build_graph(right)

        summary = tk.LabelFrame(right, text="세션 요약", bg=_BG_MID, fg="#aaa", font=("Arial", 9))
        summary.pack(fill=tk.X, pady=(8, 0))

        self._session_start_lbl = tk.Label(summary, text="시작: —", bg=_BG_MID, fg="white", font=("Arial", 9), anchor=tk.W)
        self._session_start_lbl.pack(fill=tk.X, padx=4)
        self._session_avg_lbl = tk.Label(summary, text="평균 점수: —", bg=_BG_MID, fg="white", font=("Arial", 9), anchor=tk.W)
        self._session_avg_lbl.pack(fill=tk.X, padx=4)
        self._session_alert_lbl = tk.Label(summary, text="알림 횟수: —", bg=_BG_MID, fg="white", font=("Arial", 9), anchor=tk.W)
        self._session_alert_lbl.pack(fill=tk.X, padx=4)

    def _build_graph(self, parent: tk.Frame) -> None:
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure

            fig = Figure(figsize=(4, 2.5), facecolor=_BG_MID)
            ax = fig.add_subplot(111)
            ax.set_facecolor(_BG_PANEL)
            ax.set_ylim(0, 100)
            ax.tick_params(colors="#aaa", labelsize=7)
            ax.set_ylabel("점수", color="#aaa", fontsize=7)
            ax.set_xlabel("경과(초)", color="#aaa", fontsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")
            fig.tight_layout(pad=1.5)

            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas.get_tk_widget().pack(fill=tk.X)

            self._fig = fig
            self._ax = ax
            self._canvas = canvas
            self._matplotlib_ok = True
            logger.debug("matplotlib 그래프 초기화 완료")
        except Exception:
            logger.warning("matplotlib 초기화 실패 — 그래프 비활성")
            tk.Label(parent, text="그래프 불가 (matplotlib 필요)", bg=_BG_MID, fg="#aaa").pack(pady=8)

    # ------------------------------------------------------------------
    # 갱신 루프
    # ------------------------------------------------------------------

    def _on_retry_camera(self) -> None:
        self._state.camera_retry = True
        logger.info("카메라 재시도 요청")

    def _update_frame(self) -> None:
        if not self._alive():
            return
        frame, result = self._state.read_latest()

        if not self._state.camera_ok:
            self._photo = None
            self._video_label.configure(image="", text="카메라 신호 없음")
            self._retry_btn.place(relx=0.5, rely=0.65, anchor=tk.CENTER)
        elif frame is not None:
            self._retry_btn.place_forget()
            display = frame.copy()
            # predictor가 landmarks_bgr를 제공하면 오버레이 (graceful degradation)
            if result and "landmarks_bgr" in result:
                for pt in result["landmarks_bgr"]:
                    cv2.circle(display, tuple(pt), 3, (0, 255, 0), -1)
            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((_VIDEO_W, _VIDEO_H), Image.BILINEAR)
            self._photo = ImageTk.PhotoImage(img)
            self._video_label.configure(image=self._photo, text="")
        else:
            self._retry_btn.place_forget()

        self.window.after(33, self._update_frame)

    def _update_status(self) -> None:
        if not self._alive():
            return
        _, result = self._state.read_latest()
        now = time.time()

        if self._state.paused:
            events: list[str] = []
            color = _STATUS_COLORS["paused"]
            text = "일시정지"
        elif result is None:
            events = []
            color = _STATUS_COLORS["no_camera"]
            text = "대기 중"
        else:
            events = result.get("events", [])
            if events:
                color = _STATUS_COLORS.get(events[0], "#e74c3c")
                text = " | ".join(events)
            else:
                color = _STATUS_COLORS["normal"]
                text = "정상"

        self._status_label.configure(bg=color, text=text)

        score = result.get("pose_score") if result else None
        if score is not None:
            self._score_val.configure(text=str(int(score)))
            self._score_bar["value"] = score
        else:
            self._score_val.configure(text="—")
            self._score_bar["value"] = 0

        active = set(events)
        for e in list(self._event_start):
            if e not in active:
                del self._event_start[e]
        for e in active:
            self._event_start.setdefault(e, now)

        if self._event_start:
            elapsed = int(now - min(self._event_start.values()))
            self._duration_label.configure(text=f"지속: {elapsed}초")
        else:
            self._duration_label.configure(text="지속: —")

        self.window.after(100, self._update_status)

    def _update_graph(self) -> None:
        if not self._alive() or not self._matplotlib_ok:
            return
        try:
            rows = self._conn.execute(
                "SELECT ts, score FROM score_samples "
                "WHERE session_id=? ORDER BY ts DESC LIMIT 60",
                (self._session_id,),
            ).fetchall()
        except Exception:
            logger.warning("그래프 데이터 조회 실패")
            rows = []

        xs: list[float] = []
        ys: list[float] = []
        if rows:
            rows = list(reversed(rows))
            try:
                t0 = datetime.fromisoformat(rows[0][0]).timestamp()
                xs = [datetime.fromisoformat(r[0]).timestamp() - t0 for r in rows]
                ys = [float(r[1]) for r in rows]
            except Exception:
                logger.warning("그래프 ts 파싱 실패")

        # DB flush 전 최신 점수를 마지막 포인트로 추가 (B안)
        _, result = self._state.read_latest()
        if result and (score := result.get("pose_score")) is not None:
            xs.append((xs[-1] + 1.0) if xs else 0.0)
            ys.append(float(score))

        self._ax.clear()
        self._ax.set_facecolor(_BG_PANEL)
        self._ax.set_ylim(0, 100)
        self._ax.tick_params(colors="#aaa", labelsize=7)
        self._ax.set_ylabel("점수", color="#aaa", fontsize=7)
        self._ax.set_xlabel("경과(초)", color="#aaa", fontsize=7)
        for spine in self._ax.spines.values():
            spine.set_edgecolor("#333")
        if xs:
            self._ax.plot(xs, ys, color="#3498db", linewidth=1.2)
            self._ax.fill_between(xs, ys, alpha=0.15, color="#3498db")
        self._fig.tight_layout(pad=1.5)
        self._canvas.draw()

        self.window.after(1000, self._update_graph)

    def _update_session(self) -> None:
        if not self._alive():
            return
        try:
            row = self._conn.execute(
                "SELECT started_at FROM sessions WHERE id=?",
                (self._session_id,),
            ).fetchone()
            avg = self._conn.execute(
                "SELECT AVG(score) FROM score_samples WHERE session_id=?",
                (self._session_id,),
            ).fetchone()[0]
            cnt = self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE session_id=?",
                (self._session_id,),
            ).fetchone()[0]
        except Exception:
            logger.warning("세션 요약 조회 실패")
            self.window.after(5000, self._update_session)
            return

        if row:
            try:
                started = datetime.fromisoformat(row[0]).astimezone().strftime("%H:%M:%S")
            except Exception:
                started = str(row[0])
            self._session_start_lbl.configure(text=f"시작: {started}")

        avg_text = str(int(avg)) if avg is not None else "—"
        self._session_avg_lbl.configure(text=f"평균 점수: {avg_text}")
        self._session_alert_lbl.configure(text=f"알림 횟수: {cnt}")

        self.window.after(5000, self._update_session)

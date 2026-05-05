"""plyer 기반 OS 알림 — Windows 토스트 / macOS NotificationCenter.

config.ALERT_COOLDOWN_SEC 으로 동일 알림 스팸 방지.
"""
from __future__ import annotations

import logging
import sqlite3
import time

import config
from src.utils.db import insert_event

logger = logging.getLogger(__name__)

_ALERT_MESSAGES: dict[str, tuple[str, str]] = {
    "turtle_neck":     ("자세 경고",      "목이 앞으로 나온 자세가 지속되고 있습니다."),
    "drowsy":          ("졸음 경고",      "졸음 징후가 감지되었습니다. 잠시 휴식하세요."),
    "distracted":      ("집중도 경고",    "화면 이탈 상태가 지속되고 있습니다."),
    "anomaly_posture": ("이상 자세 경고", "평소와 다른 자세 패턴이 감지되었습니다."),
}


class AlertManager:
    def __init__(self, conn: sqlite3.Connection, session_id: int) -> None:
        self._conn = conn
        self._session_id = session_id
        self._last_at: dict[str, float] = {k: 0.0 for k in _ALERT_MESSAGES}

    def set_session(self, session_id: int) -> None:
        self._session_id = session_id

    def process(self, result: dict) -> None:
        now = time.time()
        for event in result.get("events", []):
            if event not in self._last_at:
                continue
            if now - self._last_at[event] < config.ALERT_COOLDOWN_SEC:
                continue
            severity = result.get("event_severity", {}).get(event, "low")
            self._fire(event, severity, result)
            self._last_at[event] = now

    def _fire(self, event: str, severity: str, result: dict) -> None:
        title, message = _ALERT_MESSAGES[event]
        try:
            from plyer import notification
            notification.notify(title=title, message=message, timeout=5)
        except Exception:
            logger.warning("알림 전송 실패: %s / %s", title, message)

        try:
            insert_event(
                self._conn,
                session_id=self._session_id,
                event_type=event,
                severity=severity,
                score=result.get("pose_score"),
                payload=result,
            )
        except Exception:
            logger.warning("DB 이벤트 저장 실패: %s", event)

        logger.info("[ALERT] %s | %s | score=%s", event, severity, result.get("pose_score"))

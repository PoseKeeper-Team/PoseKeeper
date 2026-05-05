"""SQLite 연동 — 자세 점수, 경고 이력, 세션 통계.

WAL 모드 + 1분 단위 배치 flush 로 디스크 I/O 최소화 (config.py 참조).
스키마:
- sessions(id, started_at, ended_at, avg_score)
- events(id, session_id, ts, event_type, severity, score, payload_json)
- score_samples(id, session_id, ts, score, posture_label, focus_label)
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import config

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at   TEXT,
    avg_score  REAL
);
CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY,
    session_id   INTEGER NOT NULL,
    ts           TEXT NOT NULL,
    event_type   TEXT NOT NULL,
    severity     TEXT NOT NULL,
    score        REAL,
    payload_json TEXT
);
CREATE TABLE IF NOT EXISTS score_samples (
    id            INTEGER PRIMARY KEY,
    session_id    INTEGER NOT NULL,
    ts            TEXT NOT NULL,
    score         REAL NOT NULL,
    posture_label TEXT,
    focus_label   TEXT
);
"""


def init_db() -> sqlite3.Connection:
    path: Path = config.PATHS["db"]
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_CREATE_SQL)
    conn.commit()
    recover_open_sessions(conn)
    return conn


def recover_open_sessions(conn: sqlite3.Connection) -> None:
    """이전 비정상 종료로 닫히지 않은 세션을 현재 시각으로 종료 처리."""
    conn.execute(
        "UPDATE sessions SET ended_at=? WHERE ended_at IS NULL",
        (_now(),),
    )
    conn.commit()


def open_session(conn: sqlite3.Connection) -> int:
    cur = conn.execute(
        "INSERT INTO sessions (started_at) VALUES (?)",
        (_now(),),
    )
    conn.commit()
    return cur.lastrowid


def close_session(
    conn: sqlite3.Connection,
    session_id: int,
    avg_score: float | None,
) -> None:
    conn.execute(
        "UPDATE sessions SET ended_at=?, avg_score=? WHERE id=?",
        (_now(), avg_score, session_id),
    )
    conn.commit()


def insert_event(
    conn: sqlite3.Connection,
    session_id: int,
    event_type: str,
    severity: str,
    score: float | None,
    payload: dict | None,
) -> None:
    conn.execute(
        "INSERT INTO events "
        "(session_id, ts, event_type, severity, score, payload_json) "
        "VALUES (?,?,?,?,?,?)",
        (
            session_id,
            _now(),
            event_type,
            severity,
            score,
            json.dumps(payload) if payload else None,
        ),
    )
    conn.commit()


def flush_score_samples(
    conn: sqlite3.Connection,
    session_id: int,
    samples: list[dict],
) -> None:
    """samples: list of {"ts": str, "score": float, "posture_label": str|None, "focus_label": str|None}"""
    conn.executemany(
        "INSERT INTO score_samples "
        "(session_id, ts, score, posture_label, focus_label) "
        "VALUES (?,?,?,?,?)",
        [
            (session_id, s["ts"], s["score"], s.get("posture_label"), s.get("focus_label"))
            for s in samples
        ],
    )
    conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

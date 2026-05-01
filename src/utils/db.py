"""SQLite 연동 — 자세 점수, 경고 이력, 세션 통계.

WAL 모드 + 1분 단위 배치 flush 로 디스크 I/O 최소화 (config.py 참조).
스키마(예정):
- sessions(id, started_at, ended_at, avg_score)
- events(id, session_id, ts, event_type, severity, score)
"""

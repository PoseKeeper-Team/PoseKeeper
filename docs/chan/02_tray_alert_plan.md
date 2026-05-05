# 개발 계획 #2 — 시스템 트레이 + 알림 (`tray.py` / `alert.py` / `db.py`)

## 목표

앱을 백그라운드에서 장시간 실행하면서 위험 이벤트 발생 시 OS 알림을 보낸다.
추론 루프가 단일 웹캠 핸들을 소유하고, 대시보드와 프레임·결과를 안전하게 공유한다.
이벤트와 세션 기록을 SQLite에 저장한다.

---

## 구현할 클래스/함수 목록

| 위치 | 이름 | 역할 |
|---|---|---|
| `tray.py` | `SharedState` | 스레드 간 공유 상태 (frame, result, mode, paused) |
| `tray.py` | `should_process(mode, frame_count)` | 현재 모드·프레임 카운터로 추론 여부 판단 |
| `tray.py` | `inference_loop(state, cap, alert_manager, conn, session_id)` | 추론 스레드 |
| `tray.py` | `_build_menu(state, cmd_queue, root)` | pystray 메뉴 항목 생성 |
| `tray.py` | `_open_dashboard(state, root)` | 대시보드 창 열기 (중복 시 focus) |
| `tray.py` | `run_tray()` | 앱 진입점 |
| `alert.py` | `AlertManager(conn, session_id)` | 쿨다운 관리 + OS 알림 발생 |
| `alert.py` | `AlertManager.process(result)` | predictor 결과 받아 알림 판단 |
| `alert.py` | `AlertManager._fire(event, severity, result)` | 실제 알림 실행 + DB 저장 |
| `db.py` | `init_db()` | DB 파일·테이블 초기화, 미닫힌 세션 복구 |
| `db.py` | `open_session(conn)` | sessions 레코드 생성, session_id 반환 |
| `db.py` | `close_session(conn, session_id, avg_score)` | sessions ended_at·avg_score 업데이트 |
| `db.py` | `insert_event(conn, session_id, ...)` | events 레코드 저장 |
| `db.py` | `flush_score_samples(conn, session_id, samples)` | score_samples 배치 저장 |

---

## 스레드 구조 (수정됨)

### 핵심 제약

- `pystray.Icon.run()` 과 `tk.mainloop()` 는 둘 다 메인 스레드를 요구한다.
- 해결: **`icon.run_detached()`** 로 pystray를 백그라운드 스레드로 분리하고 메인 스레드를 Tkinter에 양보한다.
- pystray 콜백은 pystray 스레드에서 실행되므로, Tkinter 메서드를 직접 호출하면 크래시.
  → **`queue.Queue` + `root.after()` 폴링**으로 모든 Tkinter 조작을 메인 스레드에서만 실행한다.

```
main thread
  └─ tk.Tk (숨김 루트) mainloop
       └─ root.after(50ms) → cmd_queue 폴링 → Tkinter 조작 실행

pystray thread (run_detached, daemon)
  └─ 메뉴 이벤트 → cmd_queue.put(lambda: ...) 으로 명령만 전달

inference thread (daemon)
  └─ webcam read → predictor → AlertManager.process()
     └─ SharedState.update_frame()
```

---

## 상세 설계

### 1. `SharedState` (tray.py)

```python
@dataclass
class SharedState:
    mode: str = "bg"
    paused: bool = False
    running: bool = True
    latest_frame: np.ndarray | None = None
    latest_result: dict | None = None
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def update_frame(self, frame, result): ...
    def read_latest(self): ...
```

### 2. `should_process(mode, frame_count)` (tray.py)

```python
def should_process(mode: str, frame_count: int) -> bool:
    skip = config.BG_FRAME_SKIP if mode == "bg" else config.DASHBOARD_FRAME_SKIP
    return frame_count % skip == 0
```

### 3. `inference_loop` (tray.py)

```
루프:
  1. state.running False → 종료
  2. state.paused → sleep(0.1), continue
  3. cap.read() 실패 → sleep(0.1), continue
  4. frame_count 증가
  5. should_process() False → continue
  6. predictor.predict() (실패 시 stub 결과 사용)
  7. state.update_frame()
  8. alert_manager.process()
  9. score_samples 버퍼 축적 → 60초마다 flush_score_samples()
```

루프 종료 시 남은 샘플 최종 flush.

### 4. `_build_menu(state, cmd_queue, root)` (tray.py)

pystray 콜백 서명은 `(icon, item)`. icon 인자로 순환 참조 없이 `icon.stop()` 가능.

| 메뉴 | 동작 |
|---|---|
| 상태 표시 | `lambda item: _status_text(state)` (비활성) |
| 대시보드 열기 | `cmd_queue.put(lambda: _open_dashboard(state, root))` |
| 일시정지/재개 | `state.paused` 토글 |
| 설정 | stub (logger) |
| 종료 | `state.running = False`, `cmd_queue.put(root.quit)`, `icon.stop()` |

### 5. `_open_dashboard(state, root)` (tray.py)

메인 스레드에서만 호출됨 (cmd_queue 경유).

```python
def _open_dashboard(state, root):
    global _dashboard_window
    if _dashboard_window is not None:
        try:
            if _dashboard_window.winfo_exists():
                _dashboard_window.lift()
                return
        except tk.TclError:
            pass
    state.mode = "dashboard"
    win = tk.Toplevel(root)
    win.protocol("WM_DELETE_WINDOW", on_close)  # 닫힐 때 mode = "bg"
    _dashboard_window = win
```

### 6. `run_tray()` (tray.py)

```
1. pystray / PIL import 확인 — 실패 시 안내 출력 후 return
2. init_db() → conn, open_session() → session_id
3. SharedState / AlertManager(conn, session_id) 생성
4. cv2.VideoCapture(0) — 실패 시 세션 종료 후 return
5. tk.Tk() 생성, withdraw() (숨김)
6. cmd_queue 생성, root.after(50, poll_commands) 등록
7. inference_loop → daemon thread 시작
8. pystray.Icon(...).run_detached()
9. root.mainloop()   ← 메인 스레드 점유
10. (종료 finally) state.running=False, icon.stop(),
    thread.join(3), cap.release(),
    close_session(), conn.close()
```

### 7. `AlertManager` (alert.py)

```python
class AlertManager:
    def __init__(self, conn, session_id: int): ...
    def set_session(self, session_id: int): ...  # 세션 교체 용도
    def process(self, result: dict): ...
    def _fire(self, event, severity, result): ...
```

`_fire`: plyer 알림 → DB insert_event. 둘 다 try/except로 보호.

### 8. `db.py`

```python
def init_db() -> sqlite3.Connection:
    # WAL 모드, 3개 테이블 CREATE IF NOT EXISTS, recover_open_sessions() 호출

def recover_open_sessions(conn):
    # ended_at IS NULL 인 세션을 현재 시각으로 종료 처리

def open_session(conn) -> int: ...
def close_session(conn, session_id, avg_score): ...
def insert_event(conn, session_id, event_type, severity, score, payload): ...
def flush_score_samples(conn, session_id, samples): ...
```

---

## config.py 의존 항목

| 키 | 사용처 |
|---|---|
| `BG_FRAME_SKIP` | `should_process("bg", ...)` |
| `DASHBOARD_FRAME_SKIP` | `should_process("dashboard", ...)` |
| `ALERT_COOLDOWN_SEC` | AlertManager 쿨다운 |
| `PATHS["db"]` | DB 파일 경로 |

---

## 구현 순서

1. `db.py`
2. `alert.py`
3. `tray.py`
4. `main.py` 연결 및 단독 실행 확인

---

## 완료 기준

- [ ] `python main.py` 로 시스템 트레이 아이콘이 뜬다
- [ ] 메뉴 5개 항목이 표시된다 (상태 표시, 대시보드, 일시정지, 설정, 종료)
- [ ] 대시보드를 두 번 열면 기존 창이 focus된다
- [ ] 일시정지/재개가 추론 루프에 반영된다
- [ ] 동일 이벤트가 `ALERT_COOLDOWN_SEC` 안에 중복 알림되지 않는다
- [ ] 이벤트 발생 시 DB `events` 테이블에 레코드가 저장된다
- [ ] 앱 시작/종료 시 `sessions` 테이블이 생성·종료된다
- [ ] DB 파일이 없어도 자동 초기화된다
- [ ] 종료 시 웹캠 핸들이 해제되고 프로세스가 남지 않는다
- [ ] predictor stub 상태에서도 트레이가 정상 실행된다
- [ ] plyer 실패 시 앱이 죽지 않고 콘솔 로그로 fallback된다

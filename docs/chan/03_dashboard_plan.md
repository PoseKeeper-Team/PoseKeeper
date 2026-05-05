# 구현 계획: Tkinter 대시보드 UI

**참조 명세서**: `docs/05_tray_dashboard_feature_spec.md`  
**대상 파일**: `src/app/dashboard.py`, `src/app/tray.py` (일부 수정)  
**작성일**: 2026-05-05

---

## 1. 목표 요약

`tray.py`의 `_open_dashboard()` 플레이스홀더를 실제 대시보드로 교체한다.  
`SharedState`를 읽어 웹캠 영상·추론 결과를 표시하고, DB에서 통계를 불러와 그래프로 시각화한다.  
웹캠 캡처 객체는 tray가 단독 소유 — 대시보드는 읽기만 한다.

---

## 2. 클래스 설계 (`src/app/dashboard.py`)

```python
class Dashboard:
    def __init__(
        self,
        root: tk.Tk,            # 숨김 루트 (tray의 것)
        state: SharedState,     # 공유 상태 (frame / result / mode)
        conn: sqlite3.Connection,
        session_id: int,
    ): ...

    def open(self) -> None: ...   # Toplevel 생성 + 루프 시작
    def _on_close(self) -> None:  # mode → "bg", 창 파괴
```

`tray.py`의 `_open_dashboard()`가 `Dashboard(root, state, conn, session_id).open()`을 호출하도록 수정한다.

---

## 3. 레이아웃 (2-컬럼)

```
┌──────────────────────┬───────────────────────────┐
│  웹캠 영상 + 오버레이  │  시간대별 점수 그래프       │
│  (640×480 → 리사이즈) │  (matplotlib, 최근 60초)   │
├──────────────────────┼───────────────────────────┤
│  현재 상태 배지        │  세션 요약                  │
│  자세 점수 (숫자+바)   │  시작 시간 / 평균 점수      │
│  위험 지속 시간        │  알림 횟수                  │
└──────────────────────┴───────────────────────────┘
```

왼쪽 패널 `tk.Label` (이미지), 오른쪽 패널 `matplotlib FigureCanvasTkAgg`.

---

## 4. 갱신 주기 (`root.after` 기반)

| 항목 | 메서드 | 주기 |
|------|--------|-----:|
| 영상 프레임 | `_update_frame()` | 33 ms (≈30 fps) |
| 상태·점수·지속 시간 | `_update_status()` | 100 ms |
| 점수 그래프 | `_update_graph()` | 1 000 ms |
| 세션 통계 | `_update_session()` | 5 000 ms |

모두 `after()` 체인으로 메인 스레드에서만 실행 — 별도 스레드 없음.

---

## 5. 영상 렌더링 파이프라인

```
SharedState.read_latest() → (frame_bgr, result)
  → cv2.cvtColor(BGR→RGB)
  → PIL.Image.fromarray
  → PIL.Image.resize(fit within panel)
  → ImageTk.PhotoImage
  → tk.Label.configure(image=...)
```

랜드마크 오버레이: `result`에 `landmarks_bgr` 키가 있으면 `cv2.drawKeypoints` / 직접 `cv2.circle`로 점 표시.  
없으면 raw frame만 표시 (graceful degradation).

---

## 6. 점수 그래프

- `matplotlib.figure.Figure(figsize=(4, 2))` 임베드.
- `FigureCanvasTkAgg`로 Tkinter 위젯화.
- 데이터: `score_samples` 테이블에서 현재 세션의 최근 60개 행 조회.
- X축: timestamp → 경과 초, Y축: 0~100.
- 위험 이벤트 timestamp에 빨간 점 표시 (`events` 테이블 조회).
- 1초마다 `ax.clear()` → 재그리기.

---

## 7. 세션 요약 (5초 갱신)

DB 조회:
```sql
SELECT started_at, avg_score FROM sessions WHERE id = ?;
SELECT COUNT(*) FROM events WHERE session_id = ?;
SELECT AVG(score) FROM score_samples WHERE session_id = ?;
```

표시: 시작 시간, 현재 평균 점수, 누적 알림 횟수.

---

## 8. 상태 배지 색상

| 상태 | 색상 |
|------|------|
| 정상 | `#2ecc71` (초록) |
| turtle_neck | `#e67e22` (주황) |
| drowsy | `#e74c3c` (빨강) |
| distracted | `#f39c12` (노랑) |
| 일시정지 | `#95a5a6` (회색) |
| 웹캠 없음 | `#c0392b` (진빨강) |

---

## 9. 예외 처리

| 상황 | 처리 |
|------|------|
| 대시보드 중복 열기 | `_dashboard_window.lift()` (tray에 이미 구현) |
| `latest_frame is None` | "카메라 초기화 중..." 텍스트 표시 |
| DB 조회 실패 | 조용히 skip, logger.warning |
| matplotlib import 실패 | 그래프 패널 숨기고 텍스트로 대체 |

---

## 10. `tray.py` 수정 범위 (최소)

```python
# _open_dashboard 함수 내 플레이스홀더 교체
from src.app.dashboard import Dashboard

def _open_dashboard(state, root, conn, session_id):
    ...
    dashboard = Dashboard(root, state, conn, session_id)
    dashboard.open()
    _dashboard_window = dashboard.window
```

`_open_dashboard` 시그니처에 `conn`, `session_id` 추가.  
`_build_menu`의 람다도 동일하게 수정.

---

## 11. 구현 순서

1. **`dashboard.py` 골격** — `Dashboard` 클래스, `open()`, `_on_close()`, 레이아웃 프레임 생성
2. **영상 패널** — `_update_frame()` + PIL 파이프라인
3. **상태 패널** — `_update_status()` + 배지 색상
4. **그래프 패널** — matplotlib 임베드 + `_update_graph()`
5. **세션 요약 패널** — `_update_session()`
6. **tray.py 연결** — `_open_dashboard` 수정, 인자 전달
7. **예외 처리** — 웹캠 없음, matplotlib 없음 케이스
8. **런타임 검증** — `python main.py` 실행 후 열기/닫기/일시정지 시나리오 확인

---

## 12. 완료 기준 (명세서 §9 대응)

- [ ] 트레이 메뉴 "대시보드 열기" → 창이 열린다
- [ ] 대시보드 닫아도 트레이·추론이 계속 실행된다
- [ ] 대시보드 open 시 `state.mode == "dashboard"`, close 시 `"bg"`
- [ ] 웹캠 영상이 30fps 이하로 갱신된다
- [ ] 자세 점수·상태가 표시된다
- [ ] 그래프가 1초마다 갱신된다
- [ ] 세션 요약이 5초마다 갱신된다
- [ ] 웹캠 없을 때 오류 메시지가 표시되고 앱이 죽지 않는다
- [ ] 중복 열기 시 기존 창이 focus된다

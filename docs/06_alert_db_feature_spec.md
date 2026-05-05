# 기능 명세서: 알림/DB 저장

## 1. 목적

실시간 추론 결과에서 위험 상태가 일정 시간 이상 지속될 때 사용자에게 알림을 보내고, 세션과 이벤트 기록을 SQLite에 저장한다.

## 2. 대상 파일

| 파일 | 역할 |
|---|---|
| `src/app/alert.py` | OS 알림 발생, 쿨다운 관리 |
| `src/utils/db.py` | SQLite 연결, 세션/이벤트 저장 |
| `src/inference/predictor.py` | 이벤트 후보와 점수 생성 |
| `config.py` | threshold, cooldown, DB 경로 설정 |

## 3. 알림 조건

| 이벤트 | 발생 조건 | 기준 시간 |
|---|---|---:|
| 거북목 | MLP가 거북목/심한 거북목 상태를 지속 예측 | `config.TURTLE_NECK_THRESHOLD_SEC` |
| 졸음 | LSTM 졸음 또는 EAR/MAR 기준 충족 | `config.DROWSINESS_THRESHOLD_SEC` |
| 딴짓 | LSTM 딴짓 또는 HeadPose 이탈 | `config.DISTRACTION_THRESHOLD_SEC` |
| 이상 자세 | AE reconstruction error가 threshold 초과 | `config.TURTLE_NECK_THRESHOLD_SEC` |

## 4. 알림 쿨다운

동일 이벤트는 `config.ALERT_COOLDOWN_SEC` 이내에 반복 알림을 보내지 않는다.

```python
last_alert_at = {
    "turtle_neck": 0.0,
    "drowsy": 0.0,
    "distracted": 0.0,
    "anomaly_posture": 0.0,
}
```

다른 종류의 이벤트는 각각 독립적으로 쿨다운을 적용한다.

## 5. 알림 메시지

| 이벤트 | 제목 | 본문 예시 |
|---|---|---|
| `turtle_neck` | 자세 경고 | 목이 앞으로 나온 자세가 지속되고 있습니다. |
| `drowsy` | 졸음 경고 | 졸음 징후가 감지되었습니다. 잠시 휴식하세요. |
| `distracted` | 집중도 경고 | 화면 이탈 상태가 지속되고 있습니다. |
| `anomaly_posture` | 이상 자세 경고 | 평소와 다른 자세 패턴이 감지되었습니다. |

## 6. DB 스키마

### `sessions`

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `id` | INTEGER PRIMARY KEY | 세션 ID |
| `started_at` | TEXT | 세션 시작 시각 |
| `ended_at` | TEXT NULL | 세션 종료 시각 |
| `avg_score` | REAL NULL | 평균 자세 점수 |

### `events`

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `id` | INTEGER PRIMARY KEY | 이벤트 ID |
| `session_id` | INTEGER | 세션 ID |
| `ts` | TEXT | 이벤트 발생 시각 |
| `event_type` | TEXT | 이벤트 타입 |
| `severity` | TEXT | `low`, `medium`, `high` 중 하나. 기준은 `04_realtime_inference_feature_spec.md`의 severity 규칙을 따른다 |
| `score` | REAL | 당시 자세 점수 |
| `payload_json` | TEXT NULL | 저장 시점의 predictor 결과 일부 또는 전체 직렬화 |

### `score_samples`

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `id` | INTEGER PRIMARY KEY | 샘플 ID |
| `session_id` | INTEGER | 세션 ID |
| `ts` | TEXT | 기록 시각 |
| `score` | REAL | 자세 점수 |
| `posture_label` | TEXT NULL | 거북목 상태 |
| `focus_label` | TEXT NULL | 집중도 상태 |

## 7. 저장 정책

| 항목 | 정책 |
|---|---|
| SQLite 모드 | WAL 권장 |
| 이벤트 저장 | 알림 발생 시 즉시 저장 |
| 점수 샘플 | 1분 단위 batch flush 또는 일정 개수마다 저장 |
| 세션 종료 | `ended_at`, `avg_score` 업데이트 |
| DB 경로 | `config.PATHS["db"]` |

대시보드의 초단위 그래프와 현재 상태 표시는 DB가 아니라 메모리 버퍼를 우선 사용한다. DB는 영속 저장과 세션 통계 조회를 위한 저장소로 사용한다.

## 8. 예외 처리

| 상황 | 처리 |
|---|---|
| 알림 라이브러리 실패 | 콘솔 로그로 fallback |
| OS 알림 권한 없음 | 설정 안내 메시지 출력 |
| DB 파일 없음 | 상위 폴더 생성 후 DB 초기화 |
| DB lock | 짧은 재시도 후 실패 로그 |
| 앱 비정상 종료 | 다음 실행 시 열린 세션을 종료 처리 |

## 9. 완료 기준

- 동일 이벤트가 cooldown 안에서 중복 알림되지 않는다.
- 이벤트 발생 시 DB에 `events` 레코드가 저장된다.
- 앱 시작/종료 시 `sessions`가 생성/종료된다.
- 대시보드에서 최근 점수와 이벤트를 조회할 수 있다.
- DB 파일이 없어도 자동 초기화된다.

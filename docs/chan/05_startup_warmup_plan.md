# 개발 계획 #5 — 시작 워밍업 지연

## 목표

앱 시작 직후 카메라 자동 노출, 사용자 자세 정렬, 초기 랜드마크 흔들림으로 인한 오탐을 줄인다.
카메라는 즉시 열되, 설정된 워밍업 시간 동안 추론, 알림, DB 점수 저장은 수행하지 않는다.

---

## 관련 명세

| 문서 | 확인 내용 |
|---|---|
| `docs/04_realtime_inference_feature_spec.md` | 실시간 추론 흐름과 predictor 결과 구조 |
| `docs/06_alert_db_feature_spec.md` | 알림 조건과 score_samples 저장 정책 |
| `docs/07_config_environment_feature_spec.md` | 새 설정 상수 추가 원칙 |

---

## 구현 범위

| 위치 | 변경 내용 |
|---|---|
| `config.py` | `STARTUP_WARMUP_SECONDS = 3.0` 추가 |
| `src/app/tray.py` | `inference_loop()`에서 워밍업 시간 동안 추론/알림/DB 저장 skip |
| `docs/04_realtime_inference_feature_spec.md` | 시작 워밍업 정책 추가 |
| `docs/07_config_environment_feature_spec.md` | 설정값 목록에 워밍업 상수 추가 |

---

## 처리 정책

1. 앱 시작 시 웹캠은 즉시 open한다.
2. 추론 스레드 시작 시 `time.monotonic()`으로 워밍업 시작 시각을 기록한다.
3. `STARTUP_WARMUP_SECONDS` 동안은 프레임 read만 수행한다.
4. 워밍업 중에는 `predict()`, `AlertManager.process()`, `flush_score_samples()` 대상 샘플 추가를 수행하지 않는다.
5. 워밍업 시작과 종료는 로그로 남긴다.
6. 워밍업 종료 후 기존 `BG_FRAME_SKIP` / `DASHBOARD_FRAME_SKIP` 정책대로 추론한다.

---

## 검증

- `python -m pytest`
- 시작 로그에 워밍업 시작/종료가 출력되는지 확인
- 시작 직후 3초 동안 알림과 `score_samples` 저장이 발생하지 않는지 확인

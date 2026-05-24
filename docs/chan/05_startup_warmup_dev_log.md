# [Backend] 시작 워밍업 지연 적용 (완료)

## 목표
- 앱 시작 직후 3초 동안 카메라 프레임만 읽고 추론, 알림, DB 점수 저장을 지연해 초기 오탐을 줄인다.

## 완료 항목
- ✅ `config.STARTUP_WARMUP_SECONDS` 상수 추가
- ✅ `inference_loop()`에 시작 워밍업 gating 추가
- ✅ 워밍업 중 `predict()`, `AlertManager.process()`, `score_samples` 샘플 추가 skip
- ✅ 워밍업 시작/종료 로그 추가
- ✅ 실시간 추론 명세와 설정 명세 갱신
- ✅ `docs/chan/05_startup_warmup_plan.md` 구현 계획 작성

## 이슈/메모
- 워밍업 중 대시보드는 raw frame을 공유받을 수 있지만 `pose_score`는 `None`으로 둔다.
- `python -m pytest`는 수집된 테스트가 없어 exit code 1로 종료됐다.

## 다음 단계
- 실제 `python main.py` 실행 환경에서 시작 후 3초 동안 알림과 DB 점수 저장이 발생하지 않는지 런타임 확인

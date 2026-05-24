# [Backend] MLP weight 로드 호환 수정 (완료)

## 목표
- `TurtleNeckMLP`에서 학습한 `weights/mlp.pth`가 통합 앱의 `PoseMLP`로 정상 로드되도록 수정한다.

## 완료 항목
- ✅ `PoseMLP` 등록 모듈 이름을 `net`에서 `model`로 변경
- ✅ 기존 `.net` 접근 호환을 위한 property alias 유지
- ✅ `TurtleNeckMLP` 직접 import 없이 앱 표준 모델 경로 유지

## 이슈/메모
- 기존 weight의 state_dict 키는 `model.0.weight` 형식이다.
- 기존 `PoseMLP`는 `net.0.weight`를 기대해 로드 실패 가능성이 있었다.

## 다음 단계
- 실제 앱 실행 로그에서 `MLP loaded`가 출력되는지 확인
- 로드 후에도 탐지가 튀면 confidence threshold 또는 최근 N프레임 smoothing 적용 검토

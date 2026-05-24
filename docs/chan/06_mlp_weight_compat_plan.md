# 개발 계획 #6 — MLP weight 호환 수정

## 목표

`TurtleNeckMLP/train_mlp.py`로 저장된 `weights/mlp.pth`를 통합 앱의 `src.inference.predictor`에서 정상 로드할 수 있게 한다.

---

## 관련 명세

| 문서 | 확인 내용 |
|---|---|
| `docs/03_model_training_feature_spec.md` | MLP 저장 파일과 추론 로드 완료 기준 |
| `docs/04_realtime_inference_feature_spec.md` | MLP lazy-load 및 이벤트 통합 흐름 |

---

## 원인

- `weights/mlp.pth`의 `model_state_dict` 키는 `model.0.weight`, `model.3.weight` 형식이다.
- 기존 `src.models.mlp.PoseMLP`는 `self.net`으로 레이어를 등록해 `net.0.weight` 키를 기대했다.
- 레이어 구조는 같지만 등록 이름이 달라 `load_state_dict()`가 실패할 수 있다.

---

## 구현

1. `PoseMLP`의 등록 모듈 이름을 `self.model`로 변경한다.
2. 기존 코드 호환을 위해 `.net`은 읽기 전용 property alias로 유지한다.
3. `predictor.py`는 계속 `src.models.mlp.PoseMLP`만 사용한다.
4. 별도 `TurtleNeckMLP` 직접 import는 추가하지 않는다.

---

## 검증

- `python -m py_compile src/models/mlp.py src/inference/predictor.py`
- `weights/mlp.pth`를 `PoseMLP`로 실제 로드하는 smoke test

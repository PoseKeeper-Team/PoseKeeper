# 기능 명세서: 실시간 추론

## 1. 목적

웹캠 프레임에서 추출한 특징을 3개 모델에 전달하고, 사용자에게 보여줄 자세/집중도 상태와 알림 판단용 이벤트를 계산한다.

## 2. 대상 파일

| 파일 | 역할 |
|---|---|
| `src/inference/predictor.py` | 모델 lazy-load, 특징 공유, 상태 통합 |
| `src/utils/mediapipe_utils.py` | 프레임 특징 추출 |
| `src/models/*.py` | 모델 구조 정의 |
| `config.py` | weight 경로, frame skip, threshold 설정 |

## 3. 입력

| 입력 | 설명 |
|---|---|
| 웹캠 프레임 | OpenCV BGR frame |
| mode | `bg` 또는 `dashboard` |
| weight 파일 | `weights/mlp.pth`, `weights/lstm.pth`, `weights/autoencoder.pth` |
| 이전 상태 | LSTM sequence buffer, 이벤트별 지속 시간, 마지막 추론 결과 |

## 4. 출력

추론 결과는 대시보드, 알림, DB 저장이 공통으로 사용할 수 있도록 하나의 구조로 통일한다.

```python
{
    "timestamp": float,
    "posture_class": int | None,
    "posture_label": str | None,
    "posture_confidence": float | None,
    "focus_class": int | None,
    "focus_label": str | None,
    "focus_confidence": float | None,
    "anomaly_score": float | None,
    "is_anomaly": bool,
    "pose_score": float,
    "events": list[str],
    "event_severity": dict[str, str],
}
```

## 5. 상세 처리 흐름

1. 프레임을 입력받는다.
2. `should_process(mode)` 결과가 `False`면 마지막 추론 결과를 반환한다.
3. Pose landmark를 추출한다.
4. FaceMesh landmark와 EAR/MAR/HeadPose를 추출한다.
5. `build_lstm_feature()`로 `1409`차원 얼굴 피처를 결합한다.
6. MLP가 거북목 클래스를 예측한다.
7. LSTM sequence buffer가 충분히 쌓이면 집중도 클래스를 예측한다.
8. Autoencoder가 재구성 오차를 계산하고 threshold와 비교한다.
9. 각 모델 결과를 통합해 `pose_score`와 `events`를 계산한다.
10. 결과를 알림/DB/대시보드 호출부에 반환한다.

## 6. 모델 로딩 정책

| 정책 | 설명 |
|---|---|
| lazy-load | 앱 시작 직후가 아니라 첫 추론 시 weight 로드 |
| 단일 인스턴스 | 모델은 한 번만 로드하고 재사용 |
| device 반영 | `config.DEVICE` 기준으로 모델과 tensor를 같은 장치에 둠 |
| weight 없음 | 해당 모델 결과만 `None` 처리하고 앱은 계속 실행 |

LSTM 입력 피처 차원은 `config.LSTM_FEATURE_DIM`으로 고정한다. 추론 경로는 학습과 동일한 피처 순서 `face_xyz_flatten -> ear -> mar -> yaw -> pitch -> roll`를 사용해야 하며, 조합 책임은 `build_lstm_feature()` 하나에 둔다.

## 7. 상태 통합 규칙

### 이벤트 후보

| 이벤트 | 조건 |
|---|---|
| `turtle_neck` | MLP가 거북목 또는 심한 거북목을 예측 |
| `drowsy` | LSTM이 졸음을 예측하거나 EAR/MAR 기준 충족 |
| `distracted` | LSTM이 딴짓을 예측하거나 HeadPose 이탈 기준 충족 |
| `anomaly_posture` | Autoencoder reconstruction error가 threshold 초과 |

### severity 기준

알림 저장과 UI 표시에 사용할 severity는 아래 기준으로 통일한다.

| 이벤트 | low | medium | high |
|---|---|---|---|
| `turtle_neck` | `posture_label == turtle_neck` 이고 지속 시간 `< 2 * TURTLE_NECK_THRESHOLD_SEC` | `posture_label == turtle_neck` 이고 지속 시간 `>= 2 * TURTLE_NECK_THRESHOLD_SEC` 이고 `< 3 * TURTLE_NECK_THRESHOLD_SEC` | `posture_label == severe_turtle_neck` 또는 지속 시간 `>= 3 * TURTLE_NECK_THRESHOLD_SEC` |
| `drowsy` | EAR/MAR 기준 또는 LSTM `drowsy`가 임계 시간 직후 처음 충족 (`< 2 * DROWSINESS_THRESHOLD_SEC`) | 임계 시간의 2배 이상, 3배 미만 지속 | 임계 시간의 3배 이상 지속 |
| `distracted` | HeadPose 이탈 또는 LSTM `distracted`가 임계 시간 직후 처음 충족 (`< 2 * DISTRACTION_THRESHOLD_SEC`) | 임계 시간의 2배 이상, 3배 미만 지속 | 임계 시간의 3배 이상 지속 |
| `anomaly_posture` | threshold 초과 직후 첫 감지 (`< TURTLE_NECK_THRESHOLD_SEC`) | threshold 초과가 거북목 기준 시간 이상, 2배 미만 지속 | reconstruction error가 높고 지속 시간도 거북목 기준 시간의 2배 이상 |

추론 결과의 `event_severity`는 현재 프레임에서 감지된 이벤트만 포함한다. 예: `{"turtle_neck": "medium", "drowsy": "low"}`.

### 점수 계산 권장안

`pose_score`는 100점에서 위험 이벤트별 감점을 적용한다.
거북목과 심한 거북목은 둘중 하나만 적용
| 조건 | 감점 |
|---|---:|
| 거북목 | -20 |
| 심한 거북목 | -35 |
| 졸음 | -30 |
| 딴짓 | -25 |
| 이상 자세 | -15 |

최종 점수는 `0 <= pose_score <= 100` 범위로 clamp한다.

## 8. 지속 시간 기준

알림 발생은 단발 예측이 아니라 지속 시간을 기준으로 판단한다.

| 상태 | 기준 |
|---|---:|
| 거북목 | `config.TURTLE_NECK_THRESHOLD_SEC` |
| 졸음 | `config.DROWSINESS_THRESHOLD_SEC` |
| 딴짓 | `config.DISTRACTION_THRESHOLD_SEC` |
| 이상 자세 | `config.TURTLE_NECK_THRESHOLD_SEC`와 동일한 지속 시간 기준을 우선 적용 |

## 9. 예외 처리

| 상황 | 처리 |
|---|---|
| 웹캠 프레임 없음 | 이전 상태 유지 또는 빈 결과 반환 |
| Pose 미검출 | MLP/AE 결과 `None`, FaceMesh 가능하면 LSTM은 계속 |
| FaceMesh 미검출 | LSTM 결과 `None`, Pose 가능하면 MLP/AE는 계속 |
| FaceMesh는 있으나 HeadPose 계산 실패 | yaw/pitch/roll을 `0.0`으로 채우고 LSTM은 계속 |
| weight 파일 없음 | 해당 모델 비활성화 메시지 출력 |
| 모델 로드 실패 | 에러 메시지 출력 후 해당 모델 비활성화 |

## 10. 완료 기준

- weight 파일이 없어도 앱이 즉시 종료되지 않는다.
- 한 프레임에서 MediaPipe 처리를 중복 실행하지 않는다.
- MLP, LSTM, AE 결과를 하나의 결과 구조로 반환한다.
- LSTM sequence buffer가 부족할 때 명확히 `None` 상태를 반환한다.
- 알림 판단에 필요한 이벤트와 지속 시간을 계산할 수 있다.

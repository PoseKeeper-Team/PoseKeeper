# 기능 명세서: MediaPipe 랜드마크 추출

## 1. 목적

웹캠 프레임에서 자세와 얼굴 랜드마크를 추출하고, 세 모델이 공통으로 사용할 수 있는 입력 벡터와 파생 피처를 제공한다. 이 기능은 데이터 수집, 실시간 추론, 대시보드 표시에서 모두 사용된다.

## 2. 대상 파일

| 파일                                  | 역할                                                  |
| ------------------------------------- | ----------------------------------------------------- |
| `src/utils/mediapipe_utils.py`      | Pose / FaceMesh 초기화, 랜드마크 추출, 파생 피처 계산 |
| `config.py`                         | mode별 해상도, frame skip, 모델 복잡도 설정           |
| `docs/01_mediapipe_feature_plan.md` | 기존 구현 계획 참고 문서                              |

## 3. 제공 함수

| 함수                                               | 입력                       | 출력                                                       | 설명                                           |
| -------------------------------------------------- | -------------------------- | ---------------------------------------------------------- | ---------------------------------------------- |
| `should_process(mode="bg")`                      | `mode: str`              | `bool`                                                   | frame skip 기준으로 현재 프레임 처리 여부 반환 |
| `extract_pose_landmarks(frame, mode="bg")`       | BGR `np.ndarray`         | `np.ndarray(shape=(99,), dtype=float32)` 또는 `None`   | Pose 33개 점의 x, y, z 좌표 flatten            |
| `extract_face_landmarks(frame, mode="bg")`       | BGR `np.ndarray`         | `np.ndarray(shape=(1404,), dtype=float32)` 또는 `None` | FaceMesh 468개 점의 x, y, z 좌표 flatten       |
| `compute_ear(face_landmarks)`                    | FaceMesh 좌표              | `float`                                                  | 눈 감김 판단 보조값                            |
| `compute_mar(face_landmarks)`                    | FaceMesh 좌표              | `float`                                                  | 하품 판단 보조값                               |
| `estimate_head_pose(face_landmarks, image_size)` | FaceMesh 좌표, 프레임 크기 | `(yaw, pitch, roll)`                                     | 고개 방향 추정                                 |
| `build_lstm_feature(face_landmarks, image_size)` | FaceMesh 좌표, 프레임 크기 | `np.ndarray(shape=(1409,), dtype=float32)` 또는 `None` | LSTM 입력을 단일 규칙으로 조합                 |

## 4. 모드별 설정

| 모드          | 사용 상황              |                      frame skip | 해상도                                  |                           Pose complexity | FaceMesh refine                      |
| ------------- | ---------------------- | ------------------------------: | --------------------------------------- | ----------------------------------------: | ------------------------------------ |
| `bg`        | 트레이 백그라운드 추론 |        `config.BG_FRAME_SKIP` | `config.BG_CAPTURE_RESOLUTION`        |        `config.BG_MEDIAPIPE_COMPLEXITY` | `config.BG_FACEMESH_REFINE`        |
| `dashboard` | 대시보드 표시          | `config.DASHBOARD_FRAME_SKIP` | `config.DASHBOARD_CAPTURE_RESOLUTION` | `config.DASHBOARD_MEDIAPIPE_COMPLEXITY` | `config.DASHBOARD_FACEMESH_REFINE` |

## 5. 상세 처리 흐름

1. 웹캠에서 BGR 프레임을 받는다.
2. `should_process(mode)`가 `False`면 이전 추론 결과를 유지한다.
3. BGR 프레임을 RGB로 변환한다.
4. Pose 또는 FaceMesh detector를 mode별 싱글턴으로 가져온다.
5. MediaPipe `process()`를 실행한다.
6. 랜드마크가 없으면 `None`을 반환한다.
7. 랜드마크가 있으면 x, y, z 순서로 flatten하고 `float32`로 반환한다.
8. LSTM용 최종 얼굴 피처가 필요하면 `build_lstm_feature()`를 호출한다.

## 6. 데이터 계약

### Pose 입력 벡터

| 항목        | 값               |
| ----------- | ---------------- |
| landmark 수 | 33               |
| 좌표        | x, y, z          |
| shape       | `(99,)`        |
| dtype       | `np.float32`   |
| 사용 모델   | MLP, Autoencoder |

### FaceMesh 입력 벡터

| 항목        | 값                              |
| ----------- | ------------------------------- |
| landmark 수 | 468                             |
| 좌표        | x, y, z                         |
| shape       | `(1404,)`                     |
| dtype       | `np.float32`                  |
| 사용 모델   | 얼굴 파생 피처 계산의 기반 입력 |

### LSTM 얼굴 피처 벡터

| 항목      | 값                                                                   |
| --------- | -------------------------------------------------------------------- |
| 구성      | `FaceMesh(1404) + EAR(1) + MAR(1) + HeadPose(yaw, pitch, roll)(3)` |
| shape     | `(1409,)`                                                          |
| dtype     | `np.float32`                                                       |
| 피처 순서 | `face_xyz_flatten -> ear -> mar -> yaw -> pitch -> roll`           |
| 사용 모델 | LSTM                                                                 |

`extract_face_landmarks()`는 raw FaceMesh만 반환한다. LSTM 입력 결합은 반드시 `build_lstm_feature()`가 담당한다. 전처리, 학습 검증, 실시간 추론은 모두 이 함수가 만든 순서와 차원을 따른다.

## 7. 예외 처리

| 상황                                           | 처리                                                         |
| ---------------------------------------------- | ------------------------------------------------------------ |
| 프레임이 `None`                              | `None` 반환 또는 호출부에서 skip                           |
| 랜드마크 미검출                                | `None` 반환                                                |
| mode가 잘못됨                                  | `ValueError`를 발생시킨다                                  |
| MediaPipe 초기화 실패                          | 명확한 에러 메시지 출력 후 상위 호출부에 예외 전달           |
| HeadPose 계산 실패                             | `(0.0, 0.0, 0.0)`을 사용하고 failure count를 로그에 남긴다 |
| `build_lstm_feature()` 호출 시 FaceMesh 없음 | `None` 반환                                                |

## 8. 완료 기준

- `extract_pose_landmarks()`가 정상 프레임에서 shape `(99,)`를 반환한다.
- `extract_face_landmarks()`가 정상 프레임에서 shape `(1404,)`를 반환한다.
- `build_lstm_feature()`가 모든 경로에서 shape `(1409,)`와 동일 순서를 유지한다.
- 얼굴/몸이 없는 프레임에서 예외 없이 `None`을 반환한다.
- `bg`와 `dashboard` 모드에서 서로 다른 설정이 적용된다.
- 웹캠 루프에서 detector가 매 프레임 재초기화되지 않는다.
- 팀원 모델 입력에서 shape mismatch가 발생하지 않는다.

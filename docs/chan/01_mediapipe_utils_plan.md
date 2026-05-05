# 개발 계획 #1 — MediaPipe 연동 (`utils/mediapipe_utils.py`)

## 목표

웹캠 프레임에서 Pose / FaceMesh 랜드마크를 추출하고, 각 모델이 소비할 수 있는 형태로 전처리하는 공통 유틸 모듈을 완성한다.

---

## 구현할 함수 목록

| 함수 | 반환 타입 | 용도 |
|---|---|---|
| `extract_pose_landmarks(frame)` | `np.ndarray (99,)` | MLP 거북목 / Autoencoder 입력 (33 landmarks × xyz) |
| `extract_face_landmarks(frame)` | `np.ndarray (1404,)` | FaceMesh raw 좌표 (468 landmarks × xyz) |
| `compute_ear(face_landmarks)` | `float` | Eye Aspect Ratio — 졸음 감지 보조 피처 |
| `compute_mar(face_landmarks)` | `float` | Mouth Aspect Ratio — 하품 감지 보조 피처 |
| `estimate_head_pose(face_landmarks, image_size)` | `tuple[float, float, float]` | yaw, pitch, roll — 시선 이탈 감지 |
| `build_lstm_feature(face_landmarks, image_size)` | `np.ndarray (1409,)` | LSTM 입력 조합 단일 진입점 (FaceMesh 1404 + EAR 1 + MAR 1 + HeadPose 3) |

---

## 상세 설계

### 1. 모듈 레벨 초기화

```python
# config.py 설정값을 읽어 모드(백그라운드 / 대시보드)에 따라 다른 파라미터 적용
_pose_detector  = None  # mediapipe.solutions.pose.Pose
_face_detector  = None  # mediapipe.solutions.face_mesh.FaceMesh

def _get_pose(mode: str = "bg") -> mp.solutions.pose.Pose:
    """싱글턴 패턴 — 반복 초기화 비용 제거."""

def _get_face(mode: str = "bg") -> mp.solutions.face_mesh.FaceMesh:
    """싱글턴 패턴."""
```

- `mode="bg"` → `BG_MEDIAPIPE_COMPLEXITY=0`, `BG_FACEMESH_REFINE=False`
- `mode="dashboard"` → `DASHBOARD_MEDIAPIPE_COMPLEXITY=1`, `DASHBOARD_FACEMESH_REFINE=True`

### 2. `extract_pose_landmarks(frame, mode="bg")`

```
입력: BGR numpy array (웹캠 원본 프레임)
처리:
  1. BGR → RGB 변환
  2. mp.solutions.pose.Pose.process(frame_rgb)
  3. 랜드마크 없으면 None 반환
  4. landmark[i].x, .y, .z 를 순서대로 flatten → shape (99,)
출력: np.ndarray(99,) float32  /  None (랜드마크 미검출)
```

- `config.POSE_LANDMARK_DIM = 33 * 3 = 99` 와 일치해야 함
- 정규화 좌표 그대로 사용 (MediaPipe 기본값 0~1)

### 3. `extract_face_landmarks(frame, mode="bg")`

```
입력: BGR numpy array
처리:
  1. BGR → RGB 변환
  2. mp.solutions.face_mesh.FaceMesh.process(frame_rgb)
  3. 첫 번째 얼굴만 사용 (multi_face_landmarks[0])
  4. x, y, z flatten → shape (1404,)
출력: np.ndarray(1404,) float32  /  None
```

- `config.FACEMESH_LANDMARK_COUNT = 468` × 3 = 1404

### 4. `compute_ear(face_landmarks)`

EAR (Eye Aspect Ratio) 공식:

```
EAR = (||P2-P6|| + ||P3-P5||) / (2 × ||P1-P4||)
```

- 좌안 / 우안 각각 계산 후 평균
- FaceMesh landmark index (왼쪽 눈: 33, 160, 158, 133, 153, 144 / 오른쪽 눈: 362, 385, 387, 263, 373, 380)
- 반환값 < 0.2 이면 눈 감김으로 판단 (임계값은 config 에서 관리)

### 5. `compute_mar(face_landmarks)`

MAR (Mouth Aspect Ratio) 공식:

```
MAR = (||A-G|| + ||B-F|| + ||C-E||) / (2 × ||D-H||)
```

- 입술 landmark index (상: 13 / 하: 14 / 좌: 78 / 우: 308 등 FaceMesh 기준)

### 6. `estimate_head_pose(face_landmarks, image_size)`

- `image_size`: `(width, height)` — solvePnP의 픽셀 좌표 변환에 필요
- OpenCV `solvePnP` 를 이용한 yaw / pitch / roll 추정
- 3D 참조점: 코끝(1), 턱끝(152), 왼쪽 눈꼬리(33), 오른쪽 눈꼬리(263), 입 왼(78), 입 오(308) — 6점 기준
- 반환: `(yaw_deg, pitch_deg, roll_deg)` float tuple
- 계산 실패 시 `(0.0, 0.0, 0.0)` 반환 + failure count 로그 기록

### 7. `build_lstm_feature(face_landmarks, image_size)`

```
입력: FaceMesh 랜드마크, 프레임 크기 (width, height)
처리:
  1. extract_face_landmarks 로 얻은 (1404,) 벡터
  2. compute_ear() → EAR float (1)
  3. compute_mar() → MAR float (1)
  4. estimate_head_pose() → yaw, pitch, roll (3)
  5. 순서대로 concat: face_xyz_flatten -> ear -> mar -> yaw -> pitch -> roll
출력: np.ndarray(1409,) float32  /  None (FaceMesh 미검출)
```

- 피처 순서와 차원은 학습·추론 전 구간에서 동일해야 함
- LSTM 입력 조합의 단일 source of truth — 전처리(`preprocess.py`)와 실시간 추론(`predictor.py`) 모두 이 함수를 사용

---

## 예외 처리

| 상황 | 처리 |
|---|---|
| `mode`가 `"bg"` / `"dashboard"` 이외의 값 | `ValueError` 발생 |
| 프레임이 `None` | `None` 반환 |
| 랜드마크 미검출 | `None` 반환 |
| HeadPose 계산 실패 | `(0.0, 0.0, 0.0)` 반환 + failure count 로그 |
| `build_lstm_feature()` 호출 시 FaceMesh 없음 | `None` 반환 |
| MediaPipe 초기화 실패 | 명확한 에러 메시지 출력 후 상위 호출부에 예외 전달 |

---

## 프레임 스킵 처리

```python
_frame_counter: dict[str, int] = {"bg": 0, "dashboard": 0}

def should_process(mode: str = "bg") -> bool:
    """config.BG_FRAME_SKIP / DASHBOARD_FRAME_SKIP 기반으로 이번 프레임을 처리할지 결정."""
    skip = BG_FRAME_SKIP if mode == "bg" else DASHBOARD_FRAME_SKIP
    _frame_counter[mode] = (_frame_counter[mode] + 1) % (skip + 1)
    return _frame_counter[mode] == 0
```

---

## 파일 구조

```
src/utils/mediapipe_utils.py   ← 이번 작업 대상
config.py                      ← 설정값 참조 (변경 없음)
```

---

## config.py 의존 항목 (참고)

| 설정 키 | 값 | 사용처 |
|---|---|---|
| `BG_FRAME_SKIP` | 5 | 백그라운드 모드 프레임 스킵 |
| `DASHBOARD_FRAME_SKIP` | 2 | 대시보드 모드 프레임 스킵 |
| `BG_MEDIAPIPE_COMPLEXITY` | 0 | Pose 모델 경량화 |
| `DASHBOARD_MEDIAPIPE_COMPLEXITY` | 1 | Pose 모델 정밀도 향상 |
| `BG_FACEMESH_REFINE` | False | FaceMesh refine_landmarks |
| `DASHBOARD_FACEMESH_REFINE` | True | FaceMesh refine_landmarks |
| `POSE_LANDMARK_DIM` | 99 | 출력 벡터 크기 검증 |
| `FACEMESH_LANDMARK_COUNT` | 468 | 출력 벡터 크기 검증 |

---

## 팀원 연동 인터페이스

| 팀원 | 모델 | 입력 | 사용 함수 |
|---|---|---|---|
| 김민준 | MLP 거북목 탐지 | `np.ndarray (99,)` | `extract_pose_landmarks()` |
| 강지윤 | LSTM 집중도 분석 | `np.ndarray (1409,)` 시퀀스 | `build_lstm_feature()` |
| 김병훈 | Autoencoder 이상자세 | `np.ndarray (99,)` | `extract_pose_landmarks()` |

---

## 구현 순서

1. `_get_pose()` / `_get_face()` 싱글턴 초기화
2. `extract_pose_landmarks()` 구현 + 단위 테스트 (정적 이미지로 검증)
3. `extract_face_landmarks()` 구현 + 단위 테스트
4. `should_process()` 프레임 스킵 유틸 구현
5. `compute_ear()` / `compute_mar()` 구현
6. `estimate_head_pose(face_landmarks, image_size)` 구현 (solvePnP)
7. `build_lstm_feature(face_landmarks, image_size)` 구현 — 1404 + EAR + MAR + HeadPose concat
8. 전체 통합 — 웹캠 루프에서 실제 프레임으로 동작 확인

---

## 완료 기준

- [ ] `extract_pose_landmarks(frame)` → `np.ndarray(99,)` 또는 `None` 반환
- [ ] `extract_face_landmarks(frame)` → `np.ndarray(1404,)` 또는 `None` 반환
- [ ] `compute_ear()` → `float` (0.0 ~ 0.5 범위)
- [ ] `compute_mar()` → `float`
- [ ] `estimate_head_pose(face_landmarks, image_size)` → `(yaw, pitch, roll)` 각도값, 실패 시 `(0.0, 0.0, 0.0)`
- [ ] `build_lstm_feature(face_landmarks, image_size)` → `np.ndarray(1409,)` 또는 `None`, 피처 순서 고정
- [ ] `should_process(mode)` 프레임 스킵 동작 확인
- [ ] config.py 설정값 반영 확인 (mode별 complexity / refine 분기)
- [ ] 팀원 세 모델에 데이터 전달 시 shape 오류 없음

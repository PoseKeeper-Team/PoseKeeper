# 모델별 데이터 구현 확인

## 기준 파일

- 명세: `docs/02_data_feature_spec.md`, `docs/03_model_training_feature_spec.md`
- 설정 source of truth: `config.py`
- 모델: `src/models/mlp.py`, `src/models/lstm.py`, `src/models/autoencoder.py`
- MediaPipe 추출: `src/utils/mediapipe_utils.py`
- 전처리 CLI: `src/data/preprocess.py`
- 전처리 구현:
  - 공통: `src/data/preprocess_common.py`
  - MLP: `src/data/preprocess_mlp.py`
  - LSTM: `src/data/preprocess_lstm.py`
  - Autoencoder: `src/data/preprocess_ae.py`

## MLP 자세 분류

- 실제 모델: `PoseMLP`
- 목적: `normal`, `turtle_neck`, `severe_turtle_neck` 3클래스 분류
- 입력: `config.POSE_LANDMARK_DIM == 99`
- 출력: class logits `(3,)`
- 추론 경로: `predictor.py`에서 `extract_pose_landmarks()`가 만든 raw Pose 좌표 `(33 * 3)`을 그대로 입력한다.
- 데이터 전처리 결정: 로컬 mp4 분석 시 MLP 학습 데이터도 raw Pose 99차원으로 저장한다. 기존 CSV 기반 전처리는 유지하되, mp4가 있으면 mp4 경로를 우선 사용한다.
- raw mp4 위치:
  - `data/raw/posture/0_normal/*.mp4`
  - `data/raw/posture/1_turtle_neck/*.mp4`
  - `data/raw/posture/2_severe_turtle_neck/*.mp4`
  - 기존 폴더 호환: `normal`, `turtle_neck`, `severe_turtle_neck`
- processed 출력:
  - `data/processed/mlp/X.npy` shape `(N, 99)`
  - `data/processed/mlp/y.npy` shape `(N,)`
  - `data/processed/mlp/meta.json`

## LSTM 집중도 분류

- 실제 모델: `PoseLSTM`
- 목적: `normal`, `drowsy`, `distracted` 3클래스 분류
- 입력: `(sequence_length, 1409)`
- `sequence_length`: `config.LSTM_SEQUENCE_LENGTH == 30`
- `feature_dim`: `config.LSTM_FEATURE_DIM == 1409`
- 피처 순서: `face_xyz_flatten -> ear -> mar -> yaw -> pitch -> roll`
- 추론 경로: `predictor.py`에서 `process_face()` 1회 실행 후 `build_lstm_feature()`로 1409차원 벡터를 만든다.
- 데이터 전처리 결정: 로컬 mp4 분석도 같은 `build_lstm_feature()`를 사용한다.
- raw mp4 위치:
  - `data/raw/drowsiness/0_normal/*.mp4`
  - `data/raw/drowsiness/1_drowsy/*.mp4`
  - `data/raw/drowsiness/2_distracted/*.mp4`
  - 기존 폴더 호환: `data/raw/focus/focused`, `drowsy`, `distracted`
- processed 출력:
  - `data/processed/lstm/X.npy` shape `(N, sequence_length, 1409)`
  - `data/processed/lstm/y.npy` shape `(N,)`
  - `data/processed/lstm/meta.json`

## Autoencoder 이상 자세 탐지

- 실제 모델: `PoseAutoencoder`
- 목적: 정상 자세 분포에서 벗어난 이상 자세 탐지
- 입력: `config.POSE_LANDMARK_DIM == 99`
- 출력: 재구성 벡터 `(99,)`
- 학습 데이터: 정상 자세만 사용
- 추론 경로: `predictor.py`에서 Pose 99차원을 `normalize_landmarks()`로 정규화한 뒤 AE에 입력한다.
- 데이터 전처리 결정: 로컬 mp4 분석 시 AE 학습 데이터도 정상 자세 영상에서 Pose를 추출하고 `normalize_landmarks()`를 적용한다.
- raw mp4 위치:
  - `data/raw/posture/0_normal/*.mp4`
  - 기존 폴더 호환: `data/raw/posture/normal`
- processed 출력:
  - 기존 학습 스크립트 호환: `data/processed/ae/normal_poses.npy`
  - 명세 호환: `data/processed/autoencoder/X_train.npy`, `X_val.npy`
  - `data/processed/autoencoder/meta.json`

## 구현 원칙

- Google Drive는 공유 저장소로만 사용하고, 코드는 로컬 `data/raw/` 아래 mp4만 읽는다.
- 원격 Drive API, `collection://`, `page://` 경로는 전처리 입력으로 사용하지 않는다.
- 영상 분석은 `cv2.VideoCapture(video_path)`로 수행한다.
- 모든 모델 전처리는 라벨별 샘플 수, skip 프레임 수, 출력 shape를 로그로 남긴다.
- LSTM 전처리와 실시간 추론은 반드시 같은 `build_lstm_feature()`를 사용한다.
- `src/data/preprocess.py`는 CLI 진입점만 담당하고, 모델별 상세 구현은 분리된 파일에서 관리한다.

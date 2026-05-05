# PoseKeeper 기능별 상세 명세서

이 문서는 팀원이 기능 단위로 구현 범위, 입출력, 완료 기준을 빠르게 맞추기 위한 인덱스다. 각 기능 명세서는 구현 담당자가 바로 작업할 수 있도록 목적, 데이터 계약, 예외 처리, 완료 기준을 포함한다.

중요:
- 이 문서는 현재 구현 상태 설명이 아니라 협업용 목표 계약 문서다.
- 상세 알고리즘과 내부 구현은 담당자가 정하되, 외부 인터페이스와 shape 계약은 문서 기준으로 맞춘다.
- 설정값의 source of truth는 `config.py`다.

## 기능 문서 목록

| 문서 | 기능 | 주요 담당 영역 |
|---|---|---|
| [01_mediapipe_feature_spec.md](01_mediapipe_feature_spec.md) | MediaPipe 랜드마크 추출 | 웹캠 프레임에서 Pose / FaceMesh 특징 추출 |
| [02_data_feature_spec.md](02_data_feature_spec.md) | 데이터 수집/전처리/데이터셋 | raw 데이터 수집, 증강, processed 데이터 생성, Dataset/DataLoader 정의 |
| [03_model_training_feature_spec.md](03_model_training_feature_spec.md) | 모델 학습 | MLP, LSTM, Autoencoder 학습 및 weight 저장 |
| [04_realtime_inference_feature_spec.md](04_realtime_inference_feature_spec.md) | 실시간 추론 | 웹캠 입력 기반 자세/집중도 상태 판정 |
| [05_tray_dashboard_feature_spec.md](05_tray_dashboard_feature_spec.md) | 트레이 앱/대시보드 | 백그라운드 실행, 상태 표시, Tkinter UI |
| [06_alert_db_feature_spec.md](06_alert_db_feature_spec.md) | 알림/DB 저장 | 경고 발생, 쿨다운, SQLite 이벤트 저장 |
| [07_config_environment_feature_spec.md](07_config_environment_feature_spec.md) | 설정/환경 검증 | 실행 환경 감지, 경로, 요구사항 검증 |

## 공통 용어

| 용어 | 의미 |
|---|---|
| 거북목 | 목/머리 위치가 앞으로 빠진 상태. MLP 분류 모델의 주요 대상 |
| 집중도 | 정상, 졸음, 딴짓 상태를 포함하는 사용자 상태 |
| 이상 자세 | 정상 자세 분포에서 벗어난 입력. Autoencoder 재구성 오차로 판단 |
| 백그라운드 모드 | 시스템 트레이에서 장시간 실행되는 저비용 추론 모드 |
| 대시보드 모드 | 사용자가 UI를 열었을 때 더 높은 해상도와 시각화를 사용하는 모드 |
| raw 데이터 | 원본 이미지/영상 데이터 |
| processed 데이터 | 학습 가능한 형태로 정제된 데이터 |

## 전체 기능 흐름

1. `data/raw/`에 직접 수집 데이터와 공개 데이터셋을 모은다.
2. `src/data/preprocess.py`로 학습 가능한 입력 벡터/시퀀스를 만든다.
3. `src/train/train_mlp.py`, `src/train/train_lstm.py`, `src/train/train_ae.py`로 모델을 학습한다.
4. 학습된 모델은 `weights/mlp.pth`, `weights/lstm.pth`, `weights/autoencoder.pth`로 저장한다.
5. `python main.py` 실행 시 앱은 `src/app/tray.py`를 진입점으로 사용한다.
6. 백그라운드 추론 루프는 단일 웹캠 캡처를 소유하고 `src/utils/mediapipe_utils.py`로 특징을 추출한다.
7. `src/inference/predictor.py`가 세 모델의 결과를 통합해 사용자 상태를 계산한다.
8. 위험 상태가 임계 시간 이상 지속되면 `src/app/alert.py`가 알림을 발생시킨다.
9. 세션, 이벤트, 점수 기록은 `src/utils/db.py`가 SQLite에 저장한다.
10. 사용자가 트레이에서 대시보드를 열면 `src/app/dashboard.py`는 공유 프레임/결과를 읽어 영상, 점수, 그래프를 표시한다.

추가 공통 규칙:
- `LSTM` 입력은 `FaceMesh 1404 + EAR 1 + MAR 1 + HeadPose 3 = 1409` 차원으로 고정한다.
- `build_lstm_feature(...)`가 LSTM 입력 조합의 단일 source of truth다.
- raw 데이터의 source of truth는 이미지/영상 원본이며, landmark는 전처리 파생 산출물로 취급한다.
- 백그라운드와 대시보드는 단일 웹캠 캡처 루프를 공유하고, 대시보드 open 시 `dashboard` 모드로 승격한다.

## 공통 완료 기준

- 함수/클래스는 타입 힌트를 포함한다.
- 외부에서 호출할 함수는 docstring에 입력, 출력, 예외 조건을 적는다.
- 웹캠, 모델 파일, 데이터 파일이 없을 때 프로그램이 바로 죽지 않고 명확한 메시지를 낸다.
- 모델 입력 shape는 `config.py`의 상수와 일치해야 한다.
- OS별 차이가 있는 기능은 Windows CPU, Windows CUDA, macOS CPU 중 최소 1개 환경에서 동작 확인을 기록한다.
- 데이터와 weight 파일은 GitHub에 커밋하지 않고 Google Drive로 공유한다.

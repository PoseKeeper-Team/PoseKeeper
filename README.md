# PoseKeeper

웹캠 기반 자세 모니터링과 집중도 분석을 위한 팀 프로젝트다. 이 저장소의 `docs/`는 현재 구현 상태 설명이 아니라, 팀원이 기능을 나눠 구현할 때 인터페이스와 데이터 계약이 엇갈리지 않도록 맞추는 협업용 명세다.

## 저장소 구조

```text
PoseKeeper/
  config.py
  main.py
  test_env.py
  docs/
  notebooks/
  requirements/
  src/
    app/
    data/
    inference/
    models/
    train/
    utils/
```

## 기능 구성

- `src/utils/mediapipe_utils.py`: Pose / FaceMesh 추출, EAR, MAR, HeadPose, LSTM feature 조립
- `src/data/*`: 데이터 수집, 전처리, Dataset 정의
- `src/models/*`: MLP, LSTM, Autoencoder 정의
- `src/train/*`: 각 모델 학습 스크립트
- `src/inference/predictor.py`: 실시간 추론 통합
- `src/app/*`: tray, dashboard, alert
- `src/utils/db.py`: SQLite 저장

## 공통 계약

- 설정값의 source of truth는 `config.py`다.
- Pose 입력 차원은 `config.POSE_LANDMARK_DIM == 99`다.
- LSTM 입력 차원은 `config.LSTM_FEATURE_DIM == 1409`다.
- LSTM feature 순서는 `face_xyz_flatten -> ear -> mar -> yaw -> pitch -> roll`로 고정한다.
- raw 데이터의 canonical 구조는 영문 폴더명 기준이다.
- 백그라운드와 대시보드는 단일 웹캠 캡처 루프를 공유한다.

## raw 데이터 구조

```text
data/
  raw/
    posture/
      0_normal/
      1_turtle_neck/
      2_severe_turtle_neck/
    drowsiness/
      0_normal/
      1_drowsy/
      2_distracted/
  processed/
    mlp/
    lstm/
    autoencoder/
```

기존 한글 폴더 데이터가 있더라도 새 데이터는 위 영문 구조로만 추가한다. 한글 폴더 호환은 필요하면 `preprocess.py`에서만 일시적으로 처리한다.

Google Drive는 원본 영상 공유용으로만 사용한다. 각자 필요한 영상을 로컬로 다운로드한 뒤 `data/raw/drowsiness/{label}/` 아래에 배치하고, 전처리/학습 코드는 로컬 `data/raw/` 경로만 기준으로 실행한다. 예: `data/raw/drowsiness/1_drowsy/drowsy_20260513_001.mp4`

## 실행 기준

- 환경 검증: `python test_env.py`
- 앱 진입점: `python main.py`
- 학습: `python -m src.train.train_mlp`, `python -m src.train.train_lstm`, `python -m src.train.train_ae`

## 개발 환경

- 학습: Windows CUDA 또는 Google Colab
- 테스트/발표: Windows CPU 또는 macOS CPU/MPS
- 장치 선택 우선순위: CUDA -> MPS -> CPU

## 협업 규칙

- 기능 계약 변경 시 먼저 `docs/`와 `config.py`를 같이 갱신한다.
- README보다 상세한 구현 범위와 입출력 계약은 `docs/`를 기준으로 본다.
- 데이터와 weight 파일은 GitHub에 커밋하지 않고 별도 저장소로 공유한다.

## 모델별 전처리/Dataset 사용

### MLP 담당자

```powershell
python -m src.data.preprocess_mlp
```

```python
from torch.utils.data import DataLoader
from src.data.dataset_mlp import PostureDataset

dataset = PostureDataset()
loader = DataLoader(dataset, batch_size=64, shuffle=True)
features, labels = next(iter(loader))
```

저장/로드 경로: `data/processed/mlp/X.npy`, `data/processed/mlp/y.npy`

### LSTM 담당자

```powershell
python -m src.data.preprocess_lstm
```

```python
from torch.utils.data import DataLoader
from src.data.dataset_lstm import FocusSequenceDataset

dataset = FocusSequenceDataset()
loader = DataLoader(dataset, batch_size=32, shuffle=True)
sequences, labels = next(iter(loader))
```

저장/로드 경로: `data/processed/lstm/X.npy`, `data/processed/lstm/y.npy`

### Autoencoder 담당자

```powershell
python -m src.data.preprocess_ae
```

```python
from torch.utils.data import DataLoader
from src.data.dataset_ae import AutoencoderDataset

train_dataset = AutoencoderDataset(split="train")
val_dataset = AutoencoderDataset(split="val")
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
features, targets = next(iter(train_loader))
```

저장/로드 경로: `data/processed/autoencoder/X_train.npy`, `data/processed/autoencoder/X_val.npy`

공통 방식:

```python
from src.data.dataset import create_dataset

dataset = create_dataset("mlp")
dataset = create_dataset("lstm")
dataset = create_dataset("ae", split="train")
```

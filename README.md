# PoseKeeper

PoseKeeper는 웹캠으로 사용자의 자세와 집중 상태를 모니터링하는 데스크톱 앱입니다.
앱을 실행하면 백그라운드에서 카메라 프레임을 분석하고, 거북목이나 졸음 같은 상태가 일정 시간 이상 지속될 때 알림을 보냅니다. 사용자는 시스템 트레이 아이콘에서 대시보드를 열어 현재 상태, 점수 변화, 세션 기록을 확인할 수 있습니다.

## 주요 기능

- 웹캠 기반 실시간 자세 모니터링
- 거북목, 졸음, 딴짓 상태 감지
- 위험 상태 지속 시 OS 알림 표시
- 시스템 트레이 백그라운드 실행
- Tkinter 대시보드에서 현재 영상과 상태 확인
- SQLite에 세션, 경고 이벤트, 점수 기록 저장
- MLP, LSTM, Autoencoder 기반 모델 구조

## 동작 흐름

```text
앱 실행
  -> 시스템 트레이에서 백그라운드 실행
  -> 웹캠 프레임 수집
  -> MediaPipe로 Pose / FaceMesh 특징 추출
  -> 모델 추론으로 자세와 집중 상태 판단
  -> 위험 상태가 지속되면 알림 표시
  -> SQLite DB에 세션, 이벤트, 점수 저장
  -> 트레이 메뉴에서 대시보드 확인
```

백그라운드 모드는 리소스 사용을 줄이기 위해 낮은 해상도와 프레임 스킵을 사용합니다. 대시보드를 열었을 때는 사용자가 상태를 확인하기 쉽도록 더 높은 해상도의 화면 표시를 사용합니다.

## 프로젝트 구조

```text
PoseKeeper/
  config.py                 # 경로, threshold, 모델 입력 차원 등 공통 설정
  main.py                   # 앱 실행 진입점
  test_env.py               # 설치 환경 점검 스크립트
  data/
    raw/                    # 원본 영상/이미지 데이터
    processed/              # 전처리된 학습 데이터
    posture.db              # 앱 실행 중 생성되는 SQLite DB
  docs/                     # 기능별 명세서와 구현 계획
  requirements/             # 환경별 의존성 목록
  src/
    app/                    # 트레이, 대시보드, 알림
    data/                   # 데이터 수집, 전처리, Dataset
    inference/              # 실시간 추론 통합
    models/                 # MLP, LSTM, Autoencoder 모델
    train/                  # 모델 학습 스크립트
    utils/                  # MediaPipe, DB, 장치 감지 유틸
  weights/                  # 학습된 모델 파일
```

## 설치

Python 3.10 이상 환경을 권장합니다.

가상환경을 만든 뒤 프로젝트 루트에서 실행합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python setup.py
```

`setup.py`는 현재 환경을 감지해 `requirements/` 아래의 적절한 파일을 설치하고, 설치 후 `test_env.py`를 실행합니다.

환경을 직접 지정할 수도 있습니다.

```powershell
python setup.py --env cpu
python setup.py --env cuda
python setup.py --env mac
python setup.py --env colab
```

설치만 확인하고 싶을 때:

```powershell
python setup.py --no-install
python test_env.py
```

## 실행

```powershell
python main.py
```

실행 후 Windows에서는 시스템 트레이 영역에 PoseKeeper 아이콘이 표시됩니다. 트레이 메뉴에서 대시보드를 열거나 앱을 종료할 수 있습니다.

macOS에서는 메뉴바 아이콘을 확인하면 됩니다.

## 데이터베이스 확인

앱 실행 중 생성되는 기록은 SQLite 파일에 저장됩니다.

```text
data/posture.db
```

저장되는 주요 테이블은 다음과 같습니다.

| 테이블            | 설명                                    |
| ----------------- | --------------------------------------- |
| `sessions`      | 앱 실행 세션 시작/종료 시간과 평균 점수 |
| `events`        | 알림이 발생한 이벤트 기록               |
| `score_samples` | 시간별 자세/집중 점수 샘플              |

PowerShell에서 간단히 확인하려면:

```powershell
@'
import sqlite3
from pathlib import Path

p = Path("data/posture.db")
conn = sqlite3.connect(p)

print("exists:", p.exists())
print("size:", p.stat().st_size)
print("tables:", [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type=?", ("table",))])

for table in ["sessions", "events", "score_samples"]:
    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"{table}:", count)

conn.close()
'@ | python -
```

GUI로 보고 싶다면 DB Browser for SQLite 또는 VS Code의 SQLite Viewer 확장으로 `data/posture.db`를 열면 됩니다.

## 데이터와 학습 흐름

PoseKeeper의 학습 데이터 흐름은 다음과 같습니다.

```text
data/raw/
  -> preprocess
  -> data/processed/
  -> Dataset
  -> DataLoader
  -> train
  -> weights/
  -> inference
```

모델별 역할은 다음과 같습니다.

| 모델        | 대상                                | 입력                                            |
| ----------- | ----------------------------------- | ----------------------------------------------- |
| MLP         | 거북목 자세 분류                    | Pose landmark 99차원                            |
| LSTM        | 졸음/딴짓 등 집중 상태 분류         | FaceMesh + EAR + MAR + HeadPose 1409차원 시퀀스 |
| Autoencoder | 정상 자세에서 벗어난 이상 자세 탐지 | 정상 자세 Pose landmark                         |

전처리 실행 예시:

```powershell
python -m src.data.preprocess_mlp
python -m src.data.preprocess_lstm
python -m src.data.preprocess_ae
```

학습 실행 예시:

```powershell
python -m src.train.train_mlp
python -m src.train.train_lstm
python -m src.train.train_ae
```

학습된 모델 파일은 `weights/` 아래에 저장하고, 실시간 추론에서 사용합니다.

## raw 데이터 구조

새 데이터는 아래 구조를 기준으로 추가합니다.

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

Google Drive는 원본 영상 공유용으로만 사용합니다. 전처리와 학습 코드는 로컬 `data/raw/` 경로를 기준으로 실행합니다.

## 설정

공통 설정은 `config.py`에서 관리합니다.

주요 설정:

- `PATHS`: weight, raw, processed, DB 경로
- `WEIGHT_FILES`: 모델 파일 경로
- `BG_FRAME_SKIP`: 백그라운드 추론 간격
- `STARTUP_WARMUP_SECONDS`: 앱 시작 직후 추론/알림/DB 저장 지연 시간
- `ALERT_*`: 알림 threshold와 cooldown
- `POSE_LANDMARK_DIM`: MLP / Autoencoder 입력 차원
- `LSTM_FEATURE_DIM`: LSTM 입력 차원

모델 입력 shape와 경로는 `config.py`를 기준으로 맞춥니다.

## 문서

상세 구현 범위와 인터페이스 계약은 `docs/` 아래 기능별 명세서를 기준으로 합니다.

| 문서                                            | 내용                         |
| ----------------------------------------------- | ---------------------------- |
| `docs/01_mediapipe_feature_spec.md`           | MediaPipe 특징 추출          |
| `docs/02_data_feature_spec.md`                | 데이터 수집/전처리/Dataset   |
| `docs/03_model_training_feature_spec.md`      | 모델 학습                    |
| `docs/04_realtime_inference_feature_spec.md`  | 실시간 추론                  |
| `docs/05_tray_dashboard_feature_spec.md`      | 트레이 앱/대시보드           |
| `docs/06_alert_db_feature_spec.md`            | 알림/DB 저장                 |
| `docs/07_config_environment_feature_spec.md`  | 설정/환경 검증               |
| `docs/08_dataloader_training_feature_spec.md` | DataLoader와 학습 파이프라인 |

기능을 수정할 때는 README보다 `docs/`의 명세와 `config.py`를 먼저 확인합니다.

## 개발 메모

- 웹캠, 모델 파일, DB 파일이 없는 상황에서도 가능한 한 명확한 로그를 남기도록 구현합니다.
- 백그라운드 실행 최적화가 중요하므로 프레임 스킵, 낮은 해상도, `torch.no_grad()` 사용을 유지합니다.
- 로깅은 필수이며, 앱 진입점에서는 `src` 로거를 DEBUG 수준으로 설정합니다.

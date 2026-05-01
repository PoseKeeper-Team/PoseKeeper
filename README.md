# PoseKeeper

웹캠 기반 실시간 자세 모니터링 및 집중도 분석 시스템 — 백그라운드 트레이 앱.

세종대 컴공 7조 딥러닝 팀 프로젝트 (2026-1학기).

## 빠른 시작

```bash
# 1. 환경 자동 감지 + 의존성 설치
python setup.py

# 2. 설치 검증
python test_env.py

# 3. 트레이 앱 실행 (모델 학습 후)
python main.py
```

## 환경별 수동 설치

| 환경 | 명령 |
| --- | --- |
| NVIDIA GPU (CUDA) | `pip install -r requirements/cuda.txt` |
| CPU only | `pip install -r requirements/cpu.txt` |
| Mac Apple Silicon | `pip install -r requirements/mac.txt` |
| Google Colab (학습) | `!pip install -r requirements/colab.txt` |

`python setup.py --env <cuda|cpu|mac|colab>` 로 강제 지정도 가능.

### Mac MediaPipe 설치 실패 시

`setup.py` 가 자동으로 `mediapipe-silicon` 휠을 fallback 설치한다. 그래도 실패하면 Python 3.10 가상환경에서 재시도:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python setup.py --env mac
```

## 폴더 구조

```
PoseKeeper/
├── main.py              # 트레이 앱 진입점
├── config.py            # device/임계값/경로 등 모든 설정
├── setup.py             # 환경 자동 감지 + pip install
├── test_env.py          # 설치 검증 (import + 웹캠)
├── requirements/        # base / cuda / cpu / mac / colab
├── src/
│   ├── models/          # MLP / LSTM / Autoencoder 정의
│   ├── data/            # 수집·전처리·Dataset
│   ├── train/           # 학습 스크립트 3종
│   ├── inference/       # 실시간 통합 추론
│   ├── app/             # tray / dashboard / alert
│   └── utils/           # device 감지 / mediapipe 래퍼 / SQLite
├── data/                # raw / processed (gitignored)
├── weights/             # *.pth (gitignored)
├── notebooks/           # Colab 학습 노트북
└── tests/
```

## 운영 철학 — 저자원 백그라운드

상시 트레이 구동을 가정하고 CPU/RAM 사용량을 최소화한다 (`config.py` 참조):

- **프레임 스킵 5** — 30 fps 웹캠에서 6 fps 만 추론
- **MediaPipe Pose Lite** (`model_complexity=0`) + **FaceMesh refine off**
- **저해상도 캡처** (320×240) — 대시보드 열 때만 640×480 으로 승급
- **모델 lazy load** — 첫 프레임 캡처 시점에 로드
- **SQLite WAL + 1분 배치 flush** — 디스크 I/O 최소화

## 팀 / 모듈 책임

| 이름 | 모듈 |
| --- | --- |
| 김찬영 (팀장) | `models/mlp.py`, `train/train_mlp.py`, `app/tray.py`, `app/alert.py` |
| 강지윤 | `models/lstm.py`, `train/train_lstm.py`, `utils/mediapipe_utils.py` (얼굴 부분) |
| 김민준 | `app/dashboard.py`, `utils/db.py`, 시각화 |
| 김병훈 | `data/collect.py`, `data/preprocess.py`, `models/autoencoder.py`, `train/train_ae.py` |

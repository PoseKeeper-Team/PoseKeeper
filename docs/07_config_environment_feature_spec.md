# 기능 명세서: 설정/환경 검증

## 1. 목적

팀원마다 다른 실행 환경에서도 같은 명령으로 설치, 검증, 실행할 수 있게 설정값과 환경 검증 기준을 고정한다.

## 2. 대상 파일

| 파일 | 역할 |
|---|---|
| `config.py` | 전역 설정, 경로, 장치 감지 결과 |
| `src/utils/device.py` | CUDA/MPS/CPU 및 Colab 감지 |
| `test_env.py` | 설치와 실행 가능 여부 검증 |
| `requirements/*.txt` | 환경별 의존성 목록 |
| `setup.py` | 프로젝트 설치/패키징 보조 |
| `init.md` | 설치 가이드 |

## 3. 장치 감지

우선순위는 아래 순서를 따른다.

1. CUDA 사용 가능 시 `cuda`
2. Apple Silicon MPS 사용 가능 시 `mps`
3. 그 외 `cpu`

`config.DEVICE`는 모델 학습과 추론에서 동일하게 사용한다.

## 4. 환경별 요구사항

| 환경 | 용도 | requirements |
|---|---|---|
| Windows + NVIDIA GPU | 학습 | `requirements/requirements-cuda.txt` |
| Windows CPU | 테스트/발표 | `requirements/requirements-cpu.txt` |
| macOS CPU/MPS | 테스트/발표 | `requirements/requirements-cpu.txt` 또는 mac 전용 |
| Google Colab | 학습 | `requirements/colab.txt` |

## 5. 주요 설정값

| 설정 | 의미 |
|---|---|
| `BG_FRAME_SKIP` | 백그라운드 모드 추론 간격 |
| `BG_CAPTURE_RESOLUTION` | 백그라운드 웹캠 해상도 |
| `DASHBOARD_FRAME_SKIP` | 대시보드 모드 추론 간격 |
| `DASHBOARD_CAPTURE_RESOLUTION` | 대시보드 웹캠 해상도 |
| `STARTUP_WARMUP_SECONDS` | 앱 시작 후 추론/알림/DB 점수 저장을 지연할 시간 |
| `ALERT_SENSITIVITY` | 알림 민감도 |
| `ALERT_COOLDOWN_SEC` | 동일 알림 최소 간격 |
| `EAR_THRESHOLD` | 눈 감김 기준 EAR 임계값 |
| `MAR_THRESHOLD` | 하품 기준 MAR 임계값 |
| `HEADPOSE_YAW_THRESHOLD_DEG` | 좌우 시선 이탈 yaw 임계 각도 |
| `HEADPOSE_PITCH_THRESHOLD_DEG` | 상하 시선 이탈 pitch 임계 각도 |
| `POSE_LANDMARK_DIM` | Pose 모델 입력 차원 |
| `FACEMESH_LANDMARK_COUNT` | FaceMesh 랜드마크 수 |
| `LSTM_SEQUENCE_LENGTH` | LSTM 입력 시퀀스 길이 |
| `LSTM_FEATURE_DIM` | LSTM 입력 피처 차원 (`1409`) |
| `PATHS` | weight, raw, processed, DB 경로 |
| `WEIGHT_FILES` | 모델별 weight 파일 경로 |

상수 사용 원칙:
- 모델 입력 shape와 threshold 숫자의 source of truth는 `config.py`다.
- 다른 문서에는 상수명을 적고, 숫자값 설명은 `config.py` 기준으로 맞춘다.
- 새 계약 상수를 추가할 때는 `config.py`와 관련 명세서를 같은 변경으로 묶는다.

## 6. 환경 검증 항목

`python test_env.py`는 최소 아래 항목을 확인해야 한다.

| 항목 | 기대 결과 |
|---|---|
| torch import | 성공 |
| torch device | `cuda`, `mps`, `cpu` 중 하나 |
| cv2 import | 성공 |
| mediapipe import | 성공 |
| webcam | 로컬 환경에서 frame read 성공 |
| tkinter | 대시보드 환경에서 import 성공 |
| pystray | 트레이 환경에서 import 성공 |
| plyer | 알림 환경에서 import 성공 |

Colab에서는 웹캠과 트레이 관련 항목을 실패가 아니라 `SKIP`으로 처리한다.

Python 버전, 경로 준비 상태, requirements 선택은 `setup.py`와 설치 가이드에서 보조 설명할 수 있다. `test_env.py`는 실행 가능성 검증에 집중한다.

## 7. 실행 명령

```powershell
python test_env.py
python main.py
```

학습 명령은 다음을 기준으로 한다.

```powershell
python -m src.train.train_mlp
python -m src.train.train_lstm
python -m src.train.train_ae
```

## 8. 예외 처리

| 상황 | 처리 |
|---|---|
| Python 버전 불일치 | 권장 버전과 설치 링크 안내 |
| CUDA 미감지 | CPU fallback과 재설치 안내 |
| MediaPipe 설치 실패 | Python 버전 확인 메시지 출력 |
| 웹캠 실패 | 카메라 권한/점유/인덱스 안내 |
| macOS Tkinter 실패 | `python-tk` 설치 안내 |
| Colab 환경 | 웹캠/트레이 검사 skip |

## 9. 완료 기준

- 팀원이 `init.md`만 보고 환경을 구성할 수 있다.
- `python test_env.py` 결과로 실행 가능 여부를 판단할 수 있다.
- `config.py`의 경로 상수가 모든 기능에서 공통으로 사용된다.
- CUDA, MPS, CPU fallback 흐름이 명확하다.
- Colab에서는 학습만 수행하고 앱 기능은 실행하지 않는다.

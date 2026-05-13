# 기능 명세서: 모델 학습

## 1. 목적

PoseKeeper에서 사용하는 3개 모델을 각각 학습하고, 실시간 추론에서 사용할 weight 파일을 생성한다.

## 2. 대상 파일

| 파일 | 역할 |
|---|---|
| `src/models/mlp.py` | 거북목 분류 MLP 모델 |
| `src/models/lstm.py` | 집중도 분류 LSTM 모델 |
| `src/models/autoencoder.py` | 이상 자세 탐지 Autoencoder 모델 |
| `src/train/train_mlp.py` | MLP 학습 스크립트 |
| `src/train/train_lstm.py` | LSTM 학습 스크립트 |
| `src/train/train_ae.py` | Autoencoder 학습 스크립트 |
| `weights/` | 학습 완료 weight 저장 위치 |

## 3. 모델별 명세

### MLP 거북목 분류

| 항목 | 값 |
|---|---|
| 목적 | 정상/거북목/심한 거북목 3클래스 분류 |
| 입력 | Pose landmark `(99,)` |
| 출력 | class logits `(3,)` |
| 라벨 | `0: normal`, `1: turtle_neck`, `2: severe_turtle_neck` |
| 저장 파일 | `weights/mlp.pth` |
| 담당 | 김찬영 |

### LSTM 집중도 분류

| 항목 | 값 |
|---|---|
| 목적 | 정상/졸음/딴짓 3클래스 분류 |
| 입력 | `1409`차원 얼굴 피처 시계열 (`FaceMesh 1404 + EAR 1 + MAR 1 + HeadPose 3`) |
| sequence length | `config.LSTM_SEQUENCE_LENGTH` |
| 출력 | class logits `(3,)` |
| 라벨 | `0: normal`, `1: drowsy`, `2: distracted` |
| 저장 파일 | `weights/lstm.pth` |
| 담당 | 강지윤 |

### Autoencoder 이상 자세 탐지

| 항목 | 값 |
|---|---|
| 목적 | 정상 자세 분포에서 벗어난 이상 자세 탐지 |
| 입력 | Pose landmark `(99,)` |
| 출력 | 재구성 벡터 `(99,)` |
| 학습 데이터 | 정상 자세 데이터만 사용 |
| 판정 기준 | reconstruction error threshold |
| 저장 파일 | `config.WEIGHT_FILES["autoencoder"]` (`weights/autoencoder_best.pth`) |
| 담당 | 김병훈 |

## 4. 학습 스크립트 공통 요구사항

1. `config.DEVICE`를 사용해 `cuda`, `mps`, `cpu` 중 가능한 장치를 선택한다.
2. `data/processed/`에서 모델별 입력 파일을 로드한다.
3. train/validation split을 수행한다.
4. epoch별 loss와 validation metric을 출력한다.
5. 가장 좋은 validation 결과의 모델을 `weights/`에 저장한다.
6. 학습 완료 후 저장 경로와 주요 성능 지표를 출력한다.

입력 차원 검증 규칙:
- MLP/AE는 `config.POSE_LANDMARK_DIM == 99`
- LSTM은 `config.LSTM_FEATURE_DIM == 1409`

LSTM 입력 벡터는 전처리 경로와 추론 경로 모두 `build_lstm_feature()`가 만든 순서와 동일해야 한다.

## 5. 권장 CLI

```powershell
python -m src.train.train_mlp
python -m src.train.train_lstm
python -m src.train.train_ae
```

추가 옵션을 구현할 경우 아래 이름을 우선 사용한다.

| 옵션 | 설명 |
|---|---|
| `--epochs` | 학습 epoch 수 |
| `--batch-size` | batch size |
| `--lr` | learning rate |
| `--data-dir` | processed 데이터 경로 |
| `--output` | weight 저장 경로 |

## 6. 성능 지표

| 모델 | 필수 지표 | 보조 지표 |
|---|---|---|
| MLP | validation accuracy | confusion matrix, macro F1 |
| LSTM | validation accuracy | class별 recall, macro F1 |
| Autoencoder | validation reconstruction loss | threshold, false positive rate |

## 7. Weight 저장 규칙

| 파일 | 포함 내용 |
|---|---|
| `weights/mlp.pth` | model state dict, input dim, class names |
| `weights/lstm.pth` | model state dict, sequence length, input dim=`1409`, class names |
| `config.WEIGHT_FILES["autoencoder"]` (`weights/autoencoder_best.pth`) | model state dict, input dim, threshold |
| `config.WEIGHT_FILES["autoencoder_threshold"]` (`weights/threshold.npy`) | Autoencoder reconstruction error threshold |

학습 스크립트는 metadata dict 저장을 기본으로 한다. 추론 코드는 기존 단순 `state_dict` weight도 읽을 수 있게 backward compatibility를 유지한다.

## 8. 예외 처리

| 상황 | 처리 |
|---|---|
| processed 데이터 없음 | 전처리 실행 안내 메시지 출력 |
| shape 불일치 | 기대 shape와 실제 shape를 함께 출력 |
| class 불균형 심함 | 라벨별 샘플 수 출력 후 경고 |
| CUDA 메모리 부족 | batch size를 줄이라는 메시지 출력 |
| weight 저장 실패 | `weights/` 폴더 생성 후 재시도 |

## 9. 완료 기준

- 각 학습 스크립트가 단독 실행 가능하다.
- 학습 결과 weight 파일 3개와 Autoencoder threshold 파일이 `weights/`에 생성된다.
- 저장된 weight를 `src/inference/predictor.py`에서 로드할 수 있다.
- 모델 입력 dim, class 개수, sequence length가 명세와 일치한다.
- LSTM 학습과 추론이 동일한 피처 순서 `face_xyz_flatten -> ear -> mar -> yaw -> pitch -> roll`를 사용한다.
- 학습 로그에 loss와 validation 지표가 출력된다.

# 개발 계획 #4 - 모델별 Dataset 분리

## 목표

전처리 결과(`data/processed/*`)를 PyTorch 학습 스크립트에서 바로 사용할 수 있도록 모델별 Dataset 클래스를 분리한다.

## 대상 파일

| 파일 | 역할 |
|---|---|
| `src/data/dataset_mlp.py` | MLP 거북목 분류용 `PostureDataset` |
| `src/data/dataset_lstm.py` | LSTM 집중도 분류용 `FocusSequenceDataset` |
| `src/data/dataset_ae.py` | Autoencoder 이상자세용 `AutoencoderDataset` |
| `src/data/dataset.py` | 모델별 Dataset re-export 및 공통 factory |
| `docs/preprocess.md` | 전처리 후 Dataset 사용법 문서화 |

## 입력/출력 계약

| Dataset | 기본 입력 | 반환 |
|---|---|---|
| `PostureDataset` | `data/processed/mlp/X.npy`, `y.npy` | `(feature, label)` |
| `FocusSequenceDataset` | `data/processed/lstm/X.npy`, `y.npy` | `(sequence, label)` |
| `AutoencoderDataset` | `data/processed/autoencoder/X_train.npy` 또는 `X_val.npy` | `(feature, feature)` |

## 검증 규칙

- MLP/AE feature 마지막 차원은 `config.POSE_LANDMARK_DIM`과 일치해야 한다.
- LSTM feature 마지막 차원은 `config.LSTM_FEATURE_DIM`과 일치해야 한다.
- 지도학습 Dataset은 `X`와 `y`의 샘플 수가 같아야 한다.
- 파일이 없거나 shape이 맞지 않으면 경로와 기대 shape를 포함해 명확히 실패한다.

## 구현 순서

1. 모델별 Dataset 파일 생성
2. 기존 `dataset.py`에서 re-export 및 `create_dataset()` 제공
3. 문서에 Dataset 사용 예시 추가
4. `--help`가 아닌 import/간단 생성 수준의 문법 검증 수행

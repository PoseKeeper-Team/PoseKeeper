# 기능 명세서: DataLoader 및 학습 파이프라인

## 1. 목적

전처리된 `data/processed/` 산출물을 PyTorch `Dataset`/`DataLoader`로 로드하고, MLP/LSTM/Autoencoder 학습 스크립트에서 일관된 방식으로 학습, 검증, weight 저장까지 수행한다.

이 문서는 `03_model_training_feature_spec.md`의 세부 명세다. 모델 구조 자체보다 데이터 로딩, train/validation 분리, batch 구성, 학습 루프, 저장 계약을 고정하는 데 목적이 있다.

## 2. 전체 데이터 흐름

```text
data/raw/
  원본 mp4 또는 이미지
      ↓
src/data/preprocess_*.py
  MediaPipe feature 추출, 라벨 매핑, shape 검증
      ↓
data/processed/
  모델별 X.npy, y.npy, X_train.npy, X_val.npy 저장
      ↓
src/data/dataset_*.py
  PyTorch Dataset으로 감싸기
      ↓
torch.utils.data.DataLoader
  batch 단위 학습 입력 생성
      ↓
src/train/train_*.py
  학습, 검증, best checkpoint 저장
      ↓
weights/
  실시간 추론에서 로드할 .pth 저장
```

`Dataset`은 전처리를 수행하지 않는다. `Dataset`은 이미 생성된 processed 파일을 읽고, shape와 타입을 검증한 뒤 `DataLoader`가 batch를 만들 수 있게 제공하는 계층이다.

## 3. 대상 파일

| 파일                         | 역할                                                              |
| ---------------------------- | ----------------------------------------------------------------- |
| `src/data/dataset.py`      | 모델별 Dataset re-export 및 `create_dataset()` factory          |
| `src/data/dataset_mlp.py`  | `data/processed/mlp/X.npy`, `y.npy` 로드                      |
| `src/data/dataset_lstm.py` | `data/processed/lstm/X.npy`, `y.npy` 로드                     |
| `src/data/dataset_ae.py`   | `data/processed/autoencoder/X_train.npy`, `X_val.npy` 로드    |
| `src/train/train_mlp.py`   | MLP 학습, 검증,`weights/mlp.pth` 저장                           |
| `src/train/train_lstm.py`  | LSTM 학습, 검증,`weights/lstm.pth` 저장                         |
| `src/train/train_ae.py`    | Autoencoder 학습, threshold 계산, `config.WEIGHT_FILES["autoencoder"]` 저장 |
| `config.py`                | path, input dim, sequence length, device 관련 source of truth     |

## 4. 입력 데이터 계약

### MLP

| 항목             | 값                                                           |
| ---------------- | ------------------------------------------------------------ |
| 기본 경로        | `data/processed/mlp/`                                      |
| 입력 파일        | `X.npy`, `y.npy`                                         |
| X shape          | `(N, config.POSE_LANDMARK_DIM)`                            |
| 기대 feature dim | `99`                                                       |
| y shape          | `(N,)`                                                     |
| label dtype      | integer class id                                             |
| label 값         | `0: normal`, `1: turtle_neck`, `2: severe_turtle_neck` |

### LSTM

| 항목                 | 값                                                            |
| -------------------- | ------------------------------------------------------------- |
| 기본 경로            | `data/processed/lstm/`                                      |
| 입력 파일            | `X.npy`, `y.npy`                                          |
| X shape              | `(N, config.LSTM_SEQUENCE_LENGTH, config.LSTM_FEATURE_DIM)` |
| 기대 sequence length | `config.LSTM_SEQUENCE_LENGTH`                               |
| 기대 feature dim     | `1409`                                                      |
| y shape              | `(N,)`                                                      |
| label 값             | `0: normal`, `1: drowsy`, `2: distracted`               |

LSTM feature 순서는 반드시 아래 순서다.

```text
face_xyz_flatten -> ear -> mar -> yaw -> pitch -> roll
```

각 feature의 의미는 아래와 같다.

| feature | 의미 | shape |
| --- | --- | ---: |
| `face_xyz_flatten` | FaceMesh 468개 landmark의 `(x, y, z)` 좌표를 1차원으로 펼친 값 | `1404` |
| `ear` | Eye Aspect Ratio. 눈 감김 정도를 나타내는 비율 | `1` |
| `mar` | Mouth Aspect Ratio. 입 벌림/하품 정도를 나타내는 비율 | `1` |
| `yaw` | Head pose 좌우 회전 각도 | `1` |
| `pitch` | Head pose 상하 회전 각도 | `1` |
| `roll` | Head pose 기울어짐 각도 | `1` |

따라서 LSTM feature dim은 `1404 + 1 + 1 + 3 = 1409`다.

학습과 실시간 추론은 모두 같은 `build_lstm_feature()` 계약을 사용해야 한다.

### Autoencoder

| 항목             | 값                                |
| ---------------- | --------------------------------- |
| 기본 경로        | `data/processed/autoencoder/`   |
| 학습 입력        | `X_train.npy`                   |
| 검증 입력        | `X_val.npy`                     |
| X shape          | `(N, config.POSE_LANDMARK_DIM)` |
| 기대 feature dim | `99`                            |
| label            | 없음                              |
| Dataset 반환     | `(feature, feature)`            |

Autoencoder는 정상 자세만 학습한다. 라벨 파일을 요구하지 않는다.

## 5. Dataset 반환 계약

| Dataset                  | `__getitem__` 반환   | dtype                                              |
| ------------------------ | ---------------------- | -------------------------------------------------- |
| src/data/dataset_mlp.py  | `(feature, label)`   | `feature: torch.float32`, `label: torch.long`  |
| src/data/dataset_lstm.py | `(sequence, label)`  | `sequence: torch.float32`, `label: torch.long` |
| src/data/dataset_ae.py   | `(feature, feature)` | `feature: torch.float32`                         |

공통 검증 규칙:

- 파일이 없으면 누락 경로를 포함한 `FileNotFoundError`를 발생시킨다.
- `X`와 `y` 샘플 수가 다르면 기대 개수와 실제 개수를 함께 출력한다.
- feature 마지막 차원은 반드시 `config.py` 상수와 비교한다.
- LSTM은 sequence length가 다르면 현재 shape와 기대 shape를 함께 출력한다.
- `np.load(..., allow_pickle=False)`를 사용한다.

## 6. DataLoader 생성 규칙

학습 스크립트는 각 모델별 Dataset을 만든 뒤 `torch.utils.data.DataLoader`를 생성한다.

### 공통 기본값

| 옵션              |                                     기본값 | 설명                         |
| ----------------- | -----------------------------------------: | ---------------------------- |
| `--batch-size`  |                                     `64` | batch 크기                   |
| `--num-workers` |                                      `0` | Windows 호환성을 위해 기본 0 |
| `--pin-memory`  | device가 CUDA면 `True`, 아니면 `False` | GPU 전송 최적화              |
| `--drop-last`   |          train만 `True`, val은 `False` | batch norm/학습 안정성 목적  |
| `--seed`        |                                     `42` | split과 재현성 고정          |

최적화를 우선한다. 단, Windows 개발 환경에서 `num_workers > 0`은 이슈가 생길 수 있으므로 기본값은 `0`으로 둔다. 필요할 때 CLI 옵션으로만 증가시킨다.

Windows에서 `num_workers > 0`을 사용할 수 있으므로, 모든 학습 스크립트는 실행 진입점을 반드시 `if __name__ == "__main__":` guard 안에 둔다.

### MLP/LSTM split 정책

MLP와 LSTM은 하나의 processed Dataset에서 train/validation을 나눈다.

| 항목                 | 값                                     |
| -------------------- | -------------------------------------- |
| 기본 validation 비율 | `0.2`                                |
| split 방식           | `torch.utils.data.random_split`      |
| seed                 | `--seed`                             |
| train DataLoader     | `shuffle=True`, `drop_last=True`   |
| val DataLoader       | `shuffle=False`, `drop_last=False` |

클래스 불균형이 심하면 라벨별 샘플 수를 로그로 출력하고 경고한다. 첫 구현에서는 stratified split을 필수로 하지 않는다. 단, validation set에 특정 클래스가 0개가 되면 split seed를 최대 20회 재시도한다. 재시도 후에도 validation set에 특정 클래스가 0개면 학습을 중단하고 메시지를 출력한다.

MLP/LSTM은 각 클래스가 train과 validation에 최소 1개씩 들어갈 수 있어야 하므로 클래스별 최소 샘플 수는 2개다. `val_ratio` 적용 결과 `train_size < num_classes` 또는 `val_size < num_classes`이면 split을 수행하지 않고 데이터 수와 필요한 최소 수를 출력한 뒤 중단한다.

기본은 train DataLoader에서 `drop_last=True`를 사용한다. 단, smoke test나 작은 데이터셋에서 `train_size < batch_size`이면 train batch가 0개가 되므로 `drop_last=False`로 자동 전환하고 경고 로그를 남긴다.

### Autoencoder split 정책

Autoencoder는 전처리 단계에서 이미 `X_train.npy`, `X_val.npy`로 분리된 파일을 사용한다.

| 항목             | 값                                     |
| ---------------- | -------------------------------------- |
| train Dataset    | `AutoencoderDataset(split="train")`  |
| val Dataset      | `AutoencoderDataset(split="val")`    |
| train DataLoader | `shuffle=True`, `drop_last=True`   |
| val DataLoader   | `shuffle=False`, `drop_last=False` |

`train_ae.py`는 기본 입력을 `data/raw/normal_poses.npy`가 아니라 `data/processed/autoencoder/`로 맞춘다.

## 7. 학습 스크립트 CLI

### 공통 옵션

| 옵션              |                     기본값 | 설명                     |
| ----------------- | -------------------------: | ------------------------ |
| `--epochs`      | MLP `50`, LSTM `30`, Autoencoder `100` | 학습 epoch 수            |
| `--batch-size`  |                     `64` | batch size               |
| `--lr`          |                   `1e-3` | learning rate            |
| `--data-dir`    | 모델별 processed 기본 경로 | 입력 데이터 경로         |
| `--output`      |         모델별 weight 파일 | 저장할 weight 경로       |
| `--val-ratio`   |                    `0.2` | MLP/LSTM validation 비율 |
| `--seed`        |                     `42` | 재현성 seed              |
| `--num-workers` |                      `0` | DataLoader worker 수     |

모델별 `--epochs` 기본값은 아래와 같이 고정한다.

| 모델 | 기본값 |
| --- | ---: |
| MLP | `50` |
| LSTM | `30` |
| Autoencoder | `100` |

권장 실행:

```powershell
python -m src.train.train_mlp
python -m src.train.train_lstm
python -m src.train.train_ae
```

디버깅 실행:

```powershell
python -m src.train.train_mlp --epochs 3 --batch-size 16
python -m src.train.train_lstm --epochs 3 --batch-size 8
python -m src.train.train_ae --epochs 3 --batch-size 16
```

## 8. 모델별 학습 요구사항

### MLP

| 항목        | 값                            |
| ----------- | ----------------------------- |
| 모델        | `src.models.mlp`의 MLP 모델 |
| 입력 batch  | `(B, 99)`                   |
| 출력        | `(B, 3)` logits             |
| loss        | `CrossEntropyLoss`          |
| optimizer   | `Adam`                      |
| 필수 metric | validation accuracy           |
| 저장 조건   | validation accuracy 최고      |
| 저장 파일   | `weights/mlp.pth`           |

### LSTM

| 항목        | 값                              |
| ----------- | ------------------------------- |
| 모델        | `src.models.lstm`의 LSTM 모델 |
| 입력 batch  | `(B, T, 1409)`                |
| 출력        | `(B, 3)` logits               |
| loss        | `CrossEntropyLoss`            |
| optimizer   | `Adam`                        |
| 필수 metric | validation accuracy             |
| 저장 조건   | validation accuracy 최고        |
| 저장 파일   | `weights/lstm.pth`            |

LSTM은 모델 초기화 시 `input_dim=config.LSTM_FEATURE_DIM`, `sequence_length=config.LSTM_SEQUENCE_LENGTH` 계약과 맞아야 한다. 모델 생성자 인자가 다르면 학습 스크립트가 해당 모델 파일의 실제 인터페이스에 맞추되, 입력 shape 계약은 변경하지 않는다.

### Autoencoder

| 항목        | 값                                         |
| ----------- | ------------------------------------------ |
| 모델        | `src.models.autoencoder.PoseAutoencoder` |
| 입력 batch  | `(B, 99)`                                |
| 출력        | `(B, 99)`                                |
| loss        | `MSELoss`                                |
| optimizer   | `Adam`                                   |
| 필수 metric | validation reconstruction loss             |
| 저장 조건   | validation loss 최저                       |
| 저장 파일   | `config.WEIGHT_FILES["autoencoder"]` (`weights/autoencoder_best.pth`) |

학습 완료 후 validation reconstruction error 분포에서 threshold를 계산한다. 기본은 validation reconstruction error의 95 percentile이다. threshold는 `config.WEIGHT_FILES["autoencoder_threshold"]` (`weights/threshold.npy`)에도 저장해 현재 추론 코드와 호환되게 한다.

## 9. Device 및 성능 규칙

1. device는 `config.DEVICE`가 있으면 우선 사용한다.
2. `config.DEVICE`가 없거나 사용할 수 없으면 `cuda`, `mps`, `cpu` 순으로 선택한다.
3. 학습/검증 루프에서 tensor는 batch 단위로 device로 이동한다.
4. 검증은 반드시 `model.eval()`과 `torch.no_grad()`를 사용한다.
5. 학습은 불필요한 중간 tensor 저장을 피한다.
6. LSTM batch size는 메모리 사용량이 크므로 기본값을 MLP보다 낮게 조정할 수 있다.

device 선택 로직은 세 학습 스크립트에 반복 구현하지 않는다. 구현 시 `config.py`에 `get_training_device()` 같은 공통 helper를 추가하거나, 기존 `src.utils.device.detect_device()`를 감싼 공통 함수를 사용한다. `config.DEVICE`는 여전히 device source of truth다.

## 10. 로깅 규칙

로깅은 필수다. `print`만으로 끝내지 않고 가능하면 `logging`을 사용한다.

학습 시작 시 출력:

- 실행 모델명
- device
- data path
- dataset size
- train/validation size
- input shape
- class counts
- batch size
- epoch 수

epoch별 출력:

- epoch index
- train loss
- validation loss
- validation accuracy 또는 reconstruction loss
- best 갱신 여부

종료 시 출력:

- best metric
- 저장된 weight 경로
- Autoencoder threshold
- 총 학습 시간

## 11. Weight 저장 계약

`torch.save()`로 dict를 저장한다. 단순 `state_dict`만 저장하지 않는다.

### MLP

```python
{
    "model_state_dict": model.state_dict(),
    "model_name": "mlp",
    "input_dim": 99,
    "num_classes": 3,
    "class_names": ["normal", "turtle_neck", "severe_turtle_neck"],
    "best_val_accuracy": float,
    "config": {...}
}
```

### LSTM

```python
{
    "model_state_dict": model.state_dict(),
    "model_name": "lstm",
    "sequence_length": config.LSTM_SEQUENCE_LENGTH,
    "input_dim": 1409,
    "num_classes": 3,
    "class_names": ["normal", "drowsy", "distracted"],
    "best_val_accuracy": float,
    "config": {...}
}
```

LSTM도 모델 생성자 인자명과 맞춰 `input_dim`을 사용한다. `feature_dim`이라는 별도 key는 사용하지 않는다.

### Autoencoder

```python
{
    "model_state_dict": model.state_dict(),
    "model_name": "autoencoder",
    "input_dim": 99,
    "threshold": float,
    "threshold_percentile": 95.0,
    "best_val_loss": float,
    "config": {...}
}
```

추론 코드 `src/inference/predictor.py`가 기존 단순 `state_dict`를 기대하고 있으므로, 학습 스크립트 구현 시 predictor 쪽에서 dict 저장 포맷의 `model_state_dict`를 읽도록 함께 수정해야 한다. 기존 단순 `state_dict` weight도 읽을 수 있게 backward compatibility를 유지한다.

## 12. 예외 처리

| 상황                          | 처리                                            |
| ----------------------------- | ----------------------------------------------- |
| processed 파일 없음           | 실행해야 할 전처리 명령을 함께 출력             |
| feature dim 불일치            | 실제 shape와 `config.py` 기대값 출력 후 중단  |
| y 라벨 누락                   | 허용 라벨 목록과 실제 unique label 출력 후 중단 |
| train/val split 결과가 0개    | 데이터 수와 val ratio 출력 후 중단              |
| 클래스별 샘플 수 부족         | 각 클래스 최소 2개 필요 메시지 출력 후 중단     |
| train batch 생성 불가          | batch size를 낮추거나 drop_last 설정 확인 안내  |
| validation에 특정 클래스 없음 | class counts 출력 후 중단                       |
| CUDA OOM                      | batch size를 줄이라는 메시지 출력               |
| weight 저장 실패              | `weights/` 폴더 생성 후 1회 재시도            |
| 모델 생성자 인자 불일치       | 대상 모델 파일명과 필요한 인자 정보를 출력      |

## 13. 구현 순서

1. 현재 Dataset 클래스의 반환 dtype과 shape 검증을 확인한다.
2. 공통 helper를 만들지 여부를 결정한다. 중복이 작으면 각 train 파일에 직접 구현한다.
3. `train_mlp.py`에 Dataset, split, DataLoader, 학습 루프, 저장 로직을 구현한다.
4. `train_lstm.py`에 Dataset, split, DataLoader, 학습 루프, 저장 로직을 구현한다.
5. `train_ae.py`의 입력 경로를 `data/processed/autoencoder/` 기준으로 정리한다.
6. 각 스크립트에 `--epochs 1` 수준의 smoke test를 수행한다.
7. weight 저장 포맷이 predictor 로딩 코드와 맞는지 확인한다.
8. LSTM class label `0`은 학습 명세 기준 `normal`로 통일하고, UI 표시명이 필요하면 predictor에서 display label만 별도로 둔다.

## 14. 완료 기준

- `python -m src.train.train_mlp --epochs 1`이 단독 실행된다.
- `python -m src.train.train_lstm --epochs 1`이 단독 실행된다.
- `python -m src.train.train_ae --epochs 1`이 단독 실행된다.
- 세 학습 스크립트 모두 DataLoader를 통해 batch를 공급받는다.
- MLP batch shape는 `(B, 99)`다.
- LSTM batch shape는 `(B, config.LSTM_SEQUENCE_LENGTH, 1409)`다.
- Autoencoder batch shape는 `(B, 99)`이며 target도 `(B, 99)`다.
- validation metric이 epoch마다 로그에 출력된다.
- best checkpoint가 `weights/` 아래에 저장된다.
- 저장된 checkpoint에는 `model_state_dict`와 metadata가 포함된다.
- processed 데이터가 없을 때 전처리 명령 안내가 출력된다.
- shape 계약은 `config.py` 상수와 일치한다.

## 15. 현재 구현 상태 메모

- `src/data/dataset.py`와 모델별 Dataset 파일은 존재한다.
- `src.train.train_mlp`와 `src.train.train_lstm`은 학습 루프와 DataLoader 연결이 아직 필요하다.
- `src.train.train_ae`는 DataLoader 사용 코드가 있으나 기본 입력 경로와 저장 포맷을 본 명세에 맞게 정리해야 한다.

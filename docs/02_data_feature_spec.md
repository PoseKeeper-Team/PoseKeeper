# 기능 명세서: 데이터 수집/전처리/데이터셋

## 1. 목적

팀원이 수집한 웹캠 데이터와 공개 데이터셋을 학습 가능한 형태로 정리한다. 모델별 입력 형식이 다르므로 raw 데이터 구조, 라벨 규칙, 전처리 결과물을 명확히 고정한다.

## 2. 대상 파일

| 파일 | 역할 |
|---|---|
| `src/data/collect.py` | 웹캠 기반 데이터 수집 및 라벨링 도구 |
| `src/data/preprocess.py` | 모델별 전처리 CLI 진입점 |
| `src/data/preprocess_common.py` | 로컬 영상 순회, `cv2.VideoCapture`, 공통 저장 helper |
| `src/data/preprocess_mlp.py` | MLP용 posture mp4/CSV 전처리 |
| `src/data/preprocess_lstm.py` | LSTM용 drowsiness mp4/npz 전처리 |
| `src/data/preprocess_ae.py` | Autoencoder용 정상 자세 mp4/npy 전처리 |
| `src/data/dataset.py` | PyTorch Dataset / DataLoader 정의 |
| `data/raw/` | 원본 데이터 저장 위치 |
| `data/processed/` | 전처리 결과 저장 위치 |

## 3. 데이터 폴더 규칙

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

위 구조가 raw 데이터의 canonical 규칙이다. 새로 수집하는 데이터는 반드시 이 영문 폴더 구조를 사용한다. 이미 한글 폴더로 수집한 데이터가 있다면 `preprocess.py`에서만 임시 alias 매핑을 제공할 수 있다.

공식 raw 데이터의 source of truth는 이미지/영상 원본이다. landmark는 `preprocess.py`가 생성하는 파생 산출물로 취급하며, 필요할 경우 캐시로 저장할 수는 있지만 raw 데이터의 정식 포맷으로 간주하지 않는다.

Google Drive는 원본 영상 공유 저장소로만 사용한다. 각 팀원은 Drive에서 필요한 영상을 직접 로컬로 다운로드한 뒤 위 `data/raw/` 구조에 배치한다. 전처리 코드는 Google Drive API나 `collection://` 같은 원격 경로를 직접 읽지 않고, 로컬 파일 경로만 입력으로 사용한다.

drowsiness 영상 배치 예시는 다음과 같다.

```text
data/raw/drowsiness/
  0_normal/
    normal_20260513_001.mp4
  1_drowsy/
    drowsy_20260513_001.mp4
  2_distracted/
    distracted_20260513_001.mp4
```

## 4. 라벨 정의

### 거북목 라벨

| 라벨 | 이름 | 설명 |
|---:|---|---|
| 0 | normal | 정상 자세 |
| 1 | turtle_neck | 약한 거북목 |
| 2 | severe_turtle_neck | 심한 거북목 |

### 집중도 라벨

| 라벨 | 이름 | 설명 |
|---:|---|---|
| 0 | normal | 정면 주시, 눈 뜸 |
| 1 | drowsy | 눈 감김 또는 하품이 일정 시간 지속 |
| 2 | distracted | 고개/시선이 화면 밖으로 이탈 |

## 5. 데이터 수집 기능

### 입력

- 웹캠 인덱스: 기본 `0`
- 수집 라벨: posture 또는 drowsiness 기준 라벨
- 저장 간격: 프레임 단위 또는 초 단위
- 해상도: `config.TRAIN_CAPTURE_RESOLUTION`
- 공유 영상 데이터: Google Drive에서 각자 다운로드한 로컬 `.mp4` 파일

### 출력

- posture는 원본 이미지, drowsiness는 원본 영상 클립(`.mp4`)로 저장한다.
- 파일명 형식: posture는 `{label}_{timestamp}_{sample_id}.jpg`, drowsiness는 `{label}_{timestamp}_{sample_id}.mp4`
- 수집 메타데이터는 `metadata.csv`에 반드시 저장한다.

`metadata.csv` 최소 필수 컬럼:
- `sample_id`
- `task`
- `label`
- `file_path`
- `captured_at`

### 필수 동작

1. 실행 시 저장할 task와 label을 선택한다.
2. 웹캠 프레임을 화면에 표시한다.
3. 사용자가 시작/정지 키를 누르면 저장을 제어한다.
4. 저장된 샘플 수를 화면 또는 콘솔에 표시한다.
5. 종료 시 총 수집 개수를 요약한다.

## 6. 전처리 기능

### MLP 전처리

| 항목 | 값 |
|---|---|
| 입력 | posture 이미지 |
| 처리 | Pose landmark 추출, 결측 샘플 제거, 라벨 매핑 |
| 출력 | `X.npy`, `y.npy` |
| shape | `X: (N, 99)`, `y: (N,)` |

### LSTM 전처리

| 항목 | 값 |
|---|---|
| 입력 | drowsiness 영상/프레임 시퀀스 |
| 처리 | FaceMesh 1404차원 추출 후 `build_lstm_feature()`로 `1409`차원 feature sequence 구성 |
| 출력 | `X.npy`, `y.npy` |
| shape | `X: (N, sequence_length, feature_dim)`, `y: (N,)` |

`feature_dim`은 고정값 `config.LSTM_FEATURE_DIM`으로 사용한다. 피처 순서는 `face_xyz_flatten -> ear -> mar -> yaw -> pitch -> roll`로 통일하며, 조합 책임은 `build_lstm_feature()` 하나에 둔다.

### Autoencoder 전처리

| 항목 | 값 |
|---|---|
| 입력 | 정상 자세 이미지 |
| 처리 | 정상 라벨만 필터링, Pose landmark 추출 |
| 출력 | `X_train.npy`, `X_val.npy` |
| shape | `(N, 99)` |

## 7. 데이터 증강

| 증강 | 적용 대상 | 설명 |
|---|---|---|
| 좌우 반전 | 이미지 기반 데이터 | 화면 좌우 반전 상황 대응 |
| 밝기 조절 | 이미지 기반 데이터 | 조명 차이 대응 |
| 랜드마크 노이즈 | landmark 기반 데이터 | 작은 측정 오차 대응 |

증강 후에도 라벨 의미가 바뀌지 않는 경우에만 적용한다. 원본 이미지/영상에 좌우 반전을 적용한 뒤 landmark를 다시 추출하는 경우, yaw 부호와 좌우 비대칭 파생 피처가 기대대로 뒤집히는지 확인해야 한다.

## 8. Dataset / DataLoader

| Dataset | 입력 | 출력 |
|---|---|---|
| `PostureDataset` | `X: (N, 99)`, `y: (N,)` | `(feature, label)` |
| `FocusSequenceDataset` | `X: (N, T, F)`, `y: (N,)` | `(sequence, label)` |
| `AutoencoderDataset` | `X: (N, 99)` | `(feature, feature)` |

## 9. 예외 처리

| 상황 | 처리 |
|---|---|
| 웹캠 열기 실패 | 카메라 인덱스와 권한 확인 메시지 출력 |
| raw 폴더 없음 | 필요한 폴더를 생성하거나 명확한 에러 출력 |
| 라벨 폴더명 불일치 | 허용 라벨 목록을 출력 |
| drowsiness raw가 `.mp4`가 아님 | 허용 포맷 오류를 출력하고 skip |
| 랜드마크 미검출 샘플 | skip하고 skip count 기록 |
| 일부 프레임에서 HeadPose 계산 실패 | `(0, 0, 0)`으로 채우고 count 기록 |
| processed 저장 실패 | 경로와 권한 확인 메시지 출력 |

## 10. 완료 기준

- 수집 도구가 최소 1개 라벨에 대해 샘플을 저장한다.
- `preprocess.py` 실행 후 모델별 processed 파일이 생성된다.
- 전처리 결과 shape가 모델 입력 명세와 일치한다.
- LSTM 전처리 결과의 마지막 차원이 항상 `1409`다.
- 라벨별 샘플 수가 출력된다.
- 결측/skip 샘플 수가 출력된다.
- Dataset이 DataLoader에서 batch 단위로 정상 반환된다.
- `AutoencoderDataset`이 항상 `(feature, feature)` 형태를 반환한다.

# 데이터 전처리 가이드 (MP4 기반)

PoseKeeper의 3개 모델(MLP, LSTM, Autoencoder)을 학습하기 위한 **MP4 영상 전처리** 방법을 설명합니다.

---

## 📁 파일 구조

```
src/data/
├── preprocess_mlp.py        # MLP용 전처리 (거북목)
├── preprocess_lstm.py       # LSTM용 전처리 (집중도)
├── preprocess_ae.py         # Autoencoder용 전처리 (이상자세)
├── preprocess_common.py     # 공통 유틸 (MP4 읽기, feature 추출)
└── preprocess.py            # 메인 CLI 진입점
```

---

## 🎯 빠른 시작

### 0단계: 폴더 구조 준비

```
data/
└── raw/
    ├── posture/              # MLP/AE용 영상
    │   ├── 0_normal/
    │   │   ├── normal_01.mp4
    │   │   └── normal_02.mp4
    │   ├── 1_turtle_neck/
    │   │   ├── bad_posture_01.mp4
    │   │   └── bad_posture_02.mp4
    │   └── 2_severe_turtle_neck/
    │       └── severe_01.mp4
    │
    └── drowsiness/           # LSTM용 영상
        ├── 0_normal/
        │   ├── normal_01.mp4
        │   └── normal_02.mp4
        ├── 1_drowsy/
        │   ├── drowsy_01.mp4
        │   └── drowsy_02.mp4
        └── 2_distracted/
            ├── distracted_01.mp4
            └── distracted_02.mp4
```

### 1단계: 전처리 실행

```bash
# 전체 전처리 (모든 모델)
python -m src.data.preprocess --task all

# 또는 모델별 개별 실행
python -m src.data.preprocess --task mlp
python -m src.data.preprocess --task lstm
python -m src.data.preprocess --task ae
```

### 2단계: 결과 확인

```
data/processed/
├── mlp/
│   ├── X.npy        # (N, 99)
│   ├── y.npy        # (N,)
│   └── meta.json
├── lstm/
│   ├── X.npy        # (N, 30, 1409)
│   ├── y.npy        # (N,)
│   └── meta.json
└── ae/ & autoencoder/
    ├── normal_poses.npy  # (M, 99)
    ├── X_train.npy       # (M_train, 99)
    ├── X_val.npy         # (M_val, 99)
    └── meta.json
```

---

## 📊 모델별 상세 가이드

### 1️⃣ **MLP - 거북목 분류**

#### 📋 역할
정상 / 거북목 / 심한 거북목 (3클래스 분류)

#### 📂 입력 폴더 구조

```
data/raw/posture/
├── 0_normal/
│   ├── good_posture_1.mp4
│   ├── good_posture_2.mp4
│   └── ...
├── 1_turtle_neck/
│   ├── bad_posture_1.mp4
│   └── ...
└── 2_severe_turtle_neck/
    ├── severe_posture_1.mp4
    └── ...
```

**라벨 폴더명 옵션**:
- `0_normal` 또는 `normal`
- `1_turtle_neck` 또는 `turtle_neck`
- `2_severe_turtle_neck` 또는 `severe_turtle_neck`

#### ✍️ 사용법

```bash
# 기본 설정 (target_fps=6)
python -m src.data.preprocess --task mlp

# 커스텀 FPS (영상 재샘플링 속도)
python -m src.data.preprocess --task mlp --target-fps 10

# 영상당 최대 프레임 제한 (메모리 부족 시)
python -m src.data.preprocess --task mlp --max-frames-per-video 1000
```

#### 📤 출력

```
data/processed/mlp/
├── X.npy          # (N, 99) - Pose landmark 벡터
├── y.npy          # (N,) - 라벨 {0, 1, 2}
└── meta.json      # 메타정보
```

#### 📋 meta.json 구조

```json
{
  "source": "local_mp4",
  "target_fps": 6,
  "feature_dim": 99,
  "num_samples": 5234,
  "class_counts": {
    "0_normal": 2100,
    "1_turtle_neck": 2000,
    "2_severe_turtle_neck": 1134
  },
  "videos": [
    {
      "path": "data/raw/posture/0_normal/good_posture_1.mp4",
      "label": 0,
      "total_frames": 600,
      "sampled_frames": 100,
      "valid_frames": 98,
      "skipped_no_pose": 2
    }
  ]
}
```

#### 🔍 로직

```
1. data/raw/posture/ 스캔
2. 라벨 폴더별로 모든 .mp4 파일 수집
3. 각 영상에서 target_fps 기준으로 프레임 샘플링
4. MediaPipe로 Pose landmark 추출 (99차원)
5. 정규화 없이 저장 (normalize=False)
6. N개 샘플로 X.npy, y.npy 생성
```

#### 💡 팁

- **영상 품질**: 최소 480p 이상 권장
- **영상 길이**: 최소 10초 이상 권장 (샘플 수 확보)
- **FPS 설정**:
  - `target_fps=6`: 30fps 영상에서 5프레임마다 샘플링
  - `target_fps=10`: 30fps 영상에서 3프레임마다 샘플링
  - 낮을수록 샘플이 적어지지만 처리 속도 빠름

---

### 2️⃣ **LSTM - 집중도 분류**

#### 📋 역할
정상 / 졸음 / 딴짓 (3클래스 시퀀스 분류)

#### 📂 입력 폴더 구조

```
data/raw/drowsiness/
├── 0_normal/
│   ├── normal_1.mp4
│   └── normal_2.mp4
├── 1_drowsy/
│   ├── drowsy_1.mp4
│   └── drowsy_2.mp4
└── 2_distracted/
    ├── distracted_1.mp4
    └── distracted_2.mp4
```

**라벨 폴더명 옵션**:
- `0_normal`, `normal`, `focused`
- `1_drowsy`, `drowsy`
- `2_distracted`, `distracted`

#### ✍️ 사용법

```bash
# 기본 설정 (seq_len=30, stride=5, target_fps=6)
python -m src.data.preprocess --task lstm

# 커스텀 시퀀스 길이 (시간 window 변경)
# seq_len=60은 약 10초 (6fps×60/6fps)
python -m src.data.preprocess --task lstm --seq-len 60 --stride 10

# 목표 FPS 변경
python -m src.data.preprocess --task lstm --target-fps 10

# 메모리 제약 시 최대 프레임 제한
python -m src.data.preprocess --task lstm --max-frames-per-video 2000
```

#### 📤 출력

```
data/processed/lstm/
├── X.npy          # (N, 30, 1409) - 30프레임 시퀀스
├── y.npy          # (N,) - 라벨 {0, 1, 2}
└── meta.json      # 메타정보
```

#### 📋 meta.json 구조

```json
{
  "source": "local_mp4",
  "target_fps": 6,
  "seq_len": 30,
  "stride": 5,
  "feature_dim": 1409,
  "num_samples": 3458,
  "class_counts": {
    "0_normal": 1200,
    "1_drowsy": 1100,
    "2_distracted": 1158
  },
  "videos": [
    {
      "path": "data/raw/drowsiness/0_normal/normal_1.mp4",
      "label": 0,
      "total_frames": 1800,
      "sampled_frames": 300,
      "valid_frames": 298,
      "skipped_no_face": 2,
      "skipped_feature_error": 0,
      "windows": 53
    }
  ]
}
```

#### 🔍 로직

```
1. data/raw/drowsiness/ 스캔
2. 라벨 폴더별로 모든 .mp4 파일 수집
3. 각 영상에서 target_fps 기준으로 프레임 샘플링
4. MediaPipe로 FaceMesh 추출
5. 1409차원 feature 생성 (468×3 + EAR + MAR + yaw/pitch/roll)
6. 30프레임 슬라이딩 윈도우 생성 (stride=5)
   → 100프레임 영상 = 15개 시퀀스
7. (N, 30, 1409) 텐서로 저장
```

#### 💡 팁

- **seq_len (시퀀스 길이)**:
  - `30` (기본): ~5초 at 6fps
  - `60`: ~10초 at 6fps
  - LSTM이 바라보는 과거 윈도우 크기
  
- **stride (이동 간격)**:
  - `5` (기본): 겹치는 시퀀스 많음 (샘플 증가)
  - `30`: 시퀀스 안 겹침 (샘플 감소)
  
- **샘플 개수 계산**:
  ```
  1개 영상 (30fps raw 1000프레임) → target_fps=6 적용 후 약 200 feature frame
  seq_len=30, stride=5 → 약 35개 시퀀스
  3라벨 × 5영상 × 35시퀀스 = 약 525 샘플
  ```

- **최소 영상 길이**:
  - `seq_len=30` 이상이어야 최소 1개 시퀀스 생성
  - 약 5초 이상 권장

---

### 3️⃣ **Autoencoder - 이상 자세 탐지**

#### 📋 역할
정상 자세 분포만 학습 → 재구성 오차로 이상 감지 (비지도)

#### 📂 입력 폴더 구조

```
data/raw/posture/
└── 0_normal/
    ├── good_posture_1.mp4
    ├── good_posture_2.mp4
    └── ...
```

**라벨 폴더명**: `0_normal` 또는 `normal` (정상 자세만!)

#### ✍️ 사용법

```bash
# 기본 설정
python -m src.data.preprocess --task ae

# 커스텀 FPS
python -m src.data.preprocess --task ae --target-fps 10

# train/val 비율 변경 (기본 90%/10%)
python -m src.data.preprocess --task ae --ae-val-ratio 0.2
```

#### 📤 출력

```
data/processed/ae/
├── normal_poses.npy      # (M, 99) - 모든 정상 자세

data/processed/autoencoder/
├── X_train.npy           # (M_train, 99)
├── X_val.npy             # (M_val, 99)
└── meta.json
```

#### 📋 meta.json 구조

```json
{
  "source": "local_mp4",
  "target_fps": 6,
  "feature_dim": 99,
  "num_samples": 2500,
  "train_samples": 2250,
  "val_samples": 250,
  "videos": [
    {
      "path": "data/raw/posture/0_normal/good_posture_1.mp4",
      "label": 0,
      "total_frames": 600,
      "sampled_frames": 100,
      "valid_frames": 98,
      "skipped_no_pose": 2
    }
  ]
}
```

#### 🔍 로직

```
1. data/raw/posture/0_normal/ 만 스캔 (정상 자세만!)
2. 모든 .mp4 파일에서 프레임 샘플링
3. MediaPipe로 Pose landmark 추출
4. normalize_landmarks() 적용 (정규화 필수)
5. train/val 분할 (기본 90%/10%)
6. 각각 X_train.npy, X_val.npy 저장
```

#### 💡 팁

- **정상 자세만 필요**: 비지도 학습 (이상 라벨 불필요)
- **최소 샘플 수**: 1000개 이상 권장
- **val_ratio**: 검증 데이터 비율
  - `0.1` (기본): 90% train, 10% val
  - `0.2`: 80% train, 20% val
  - 작을수록 학습 데이터 많음, 클수록 검증 정확도 높음
- **정규화**: AE만 유일하게 `normalize=True` 적용

---

## 🛠️ CLI 옵션 전체

### 공통 옵션

```bash
python -m src.data.preprocess --task {mlp|lstm|ae|all} [options]
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--task` | 필수 | 전처리 대상: `mlp`, `lstm`, `ae`, `all` |
| `--target-fps` | 6 | 영상 샘플링 FPS |
| `--max-frames-per-video` | None | 영상당 최대 유효 feature 수 (메모리 제약 시) |

### LSTM 전용

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--seq-len` | 30 | 시퀀스 길이 (프레임) |
| `--stride` | 5 | 슬라이딩 윈도우 이동 간격 |

### Autoencoder 전용

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--ae-val-ratio` | 0.1 | 검증 데이터 비율 |

---

## 📊 전처리 파이프라인 이해

### 1. 영상 읽기 & FPS 조정

```python
# src/data/preprocess_common.py
def extract_pose_video_features(..., target_fps=6):
    """
    영상 FPS → target_fps로 재샘플링
    
    예: 30fps 영상 → target_fps=6
    → step = 30/6 = 5
    → 5프레임마다 1프레임 샘플링
    """
```

### 2. Feature 추출

**MLP/AE**:
```
Frame → MediaPipe Pose → 33 landmarks × 3 coords = 99차원
```

**LSTM**:
```
Frame → MediaPipe FaceMesh → 468 landmarks × 3 coords = 1404
      → Eye Aspect Ratio (EAR) = 1
      → Mouth Aspect Ratio (MAR) = 1
      → Head Pose (yaw, pitch, roll) = 3
      ─────────────────────────────────────────
      합계 = 1409차원
```

### 3. 정규화 (AE만)

```python
# AE 입력 정규화: 양 어깨 중심을 원점으로 이동하고 어깨 너비로 스케일링
# MLP는 정규화 안 함
normalized = normalize_landmarks(pose_vec)
```

### 4. 저장

```
MLP: (N, 99) → X.npy
     (N,) → y.npy
     
LSTM: (N, 30, 1409) → X.npy
      (N,) → y.npy
      
AE: (M, 99) → normal_poses.npy
    (M_train, 99) → X_train.npy
    (M_val, 99) → X_val.npy
```

---

## 🐛 트러블슈팅

### 문제: "영상 파일을 열 수 없습니다"

```
RuntimeError: 영상 파일을 열 수 없습니다: ...
```

**원인**: 파일 경로 오류 또는 영상 코덱 미지원  
**해결**:
```bash
# 경로 확인
ls data/raw/posture/0_normal/

# 영상 포맷 확인 (ffprobe 필요)
ffprobe your_video.mp4

# 지원 형식: .mp4, .avi, .mov, .mkv
```

---

### 문제: "Pose 미검출 / Face 미검출"

```
[MLP] 유효 Pose 없음: ...
[LSTM] 너무 짧음: ...
```

**원인**: 조명 나쁨 / 각도 어려움 / 영상 품질 낮음  
**해결**:
```bash
# 1. 영상 품질 확인 (480p 이상 권장)
# 2. 밝기와 각도 조정
# 3. target_fps를 높여서 더 자주 샘플링
python -m src.data.preprocess --task mlp --target-fps 10

# 4. 더 많은 영상 추가
```

---

### 문제: "mp4에서 추출된 샘플이 없습니다"

```
RuntimeError: [MLP] mp4에서 추출된 샘플이 없습니다.
```

**원인**: 모든 영상에서 feature 추출 실패  
**해결**:
```bash
# 1. 폴더 구조 확인
data/raw/posture/
  ├── 0_normal/     ← 이 폴더 반드시 필요!
  ├── 1_turtle_neck/
  └── 2_severe_turtle_neck/

# 2. 각 폴더에 최소 1개 .mp4 파일 배치
# 3. 로그 확인 (상세 에러 메시지)
python -m src.data.preprocess --task mlp 2>&1 | tee preprocess.log
```

---

### 문제: "메모리 부족"

```
MemoryError: ...
```

**원인**: 너무 긴 영상 또는 너무 많은 영상  
**해결**:
```bash
# 영상당 최대 프레임 제한
python -m src.data.preprocess --task lstm --max-frames-per-video 500

# target_fps 낮춰서 샘플링 적게
python -m src.data.preprocess --task mlp --target-fps 3
```

---

## ✅ 체크리스트

### 전처리 전

- [ ] 폴더 구조 정확한지 확인
- [ ] 각 라벨 폴더에 최소 1개 영상 배치
- [ ] 영상 품질 480p 이상
- [ ] 영상 길이 최소 10초 이상

### 전처리 후

**MLP**:
- [ ] `data/processed/mlp/X.npy` 존재
- [ ] `data/processed/mlp/y.npy` 존재
- [ ] `data/processed/mlp/meta.json` 존재
- [ ] X shape: `(N, 99)`
- [ ] y 라벨: `{0, 1, 2}` 포함
- [ ] N > 1000 (샘플 충분한지 확인)

**LSTM**:
- [ ] `data/processed/lstm/X.npy` 존재
- [ ] `data/processed/lstm/y.npy` 존재
- [ ] `data/processed/lstm/meta.json` 존재
- [ ] X shape: `(N, 30, 1409)`
- [ ] y 라벨: `{0, 1, 2}` 모두 포함
- [ ] 각 클래스 N > 100

**Autoencoder**:
- [ ] `data/processed/ae/normal_poses.npy` 존재
- [ ] `data/processed/autoencoder/X_train.npy` 존재
- [ ] `data/processed/autoencoder/X_val.npy` 존재
- [ ] shape: `(*, 99)`
- [ ] 샘플 수 > 1000

---

## 🔗 다음 단계

전처리 완료 후 학습 시작:

> 참고: 현재 `src.train.train_mlp`, `src.train.train_lstm`은 학습 스크립트 파일만 있고 구현이 비어 있습니다. 아래 명령은 해당 스크립트 구현 후 사용하세요.

```bash
# MLP 학습
python -m src.train.train_mlp

# LSTM 학습
python -m src.train.train_lstm

# Autoencoder 학습
python -m src.train.train_ae
```

---

## 💡 Python API 직접 사용

```python
from src.data.preprocess_mlp import preprocess_mlp
from src.data.preprocess_lstm import preprocess_lstm_focus
from src.data.preprocess_ae import preprocess_ae

# MLP 전처리
preprocess_mlp(target_fps=6, max_frames_per_video=1000)

# LSTM 전처리
preprocess_lstm_focus(seq_len=30, stride=5, target_fps=6)

# AE 전처리
preprocess_ae(target_fps=6, val_ratio=0.1)
```

---

## 📞 담당자

- **MLP**: 찬영 (preprocess_mlp.py)
- **LSTM**: 지윤 (preprocess_lstm.py)
- **Autoencoder**: 병훈 (preprocess_ae.py)
- **공통**: 모두 (preprocess_common.py)

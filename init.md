# PoseKeeper 설치 가이드

팀원이 처음 받았을 때 끝까지 따라 하면 동작하도록 작성.

---

## A. Windows (NVIDIA GPU — 학습용)

> **Python 3.11** 필요 (3.11.x 권장, 3.12 이상은 mediapipe 미지원) 근데 3.8~3.11까지 가능
> [python.org/downloads](https://www.python.org/downloads/release/python-3119/) 에서 설치 후 진행.

### A-1. 프로젝트 클론 + venv

```powershell
cd C:\Users\<사용자이름>\Documents   # 원하는 경로로 변경
# git clone <팀 레포 URL> PoseKeeper
cd PoseKeeper

python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> PowerShell 실행정책 에러가 뜨면:
>
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```
>
> 한 번만 실행하고 다시 `Activate.ps1`.

### A-2. 의존성 설치

```powershell
python -m pip install --upgrade pip

# GPU 있는 경우 (학습용)
pip install -r requirements/requirements-cuda.txt

# GPU 없는 경우 (테스트용)
pip install -r requirements/requirements-cpu.txt
```

### A-3. 환경 검증

```powershell
python test_env.py
```

**기대 결과 (GPU)**:

```
[OK]   torch device=cuda — cuda_available=True ...
[OK]   webcam VideoCapture(0) — frame 640x480
[Summary] 12/12 passed, 0 failed, 0 skipped
```

---

## B. macOS (Apple Silicon / Intel — 테스트용)

> **Python 3.11** 필요. `python3 --version`으로 확인 후 다르면 [python.org](https://www.python.org/downloads/release/python-3119/) 또는 `brew install python@3.11`. 근데 3.8~3.11까지 가능

### B-1. 프로젝트 + venv

```bash
cd ~/Projects   # 또는 원하는 경로
# git clone <팀 레포 URL> PoseKeeper
cd PoseKeeper

python3 -m venv .venv
source .venv/bin/activate
```

### B-2. 의존성 설치

```bash
python -m pip install --upgrade pip

# GPU 없는 경우
pip install -r requirements/requirements-cpu.txt
```

### B-3. MediaPipe 설치 실패 시 (Apple Silicon에서 자주 있음)

```bash
pip install mediapipe-silicon
```

### B-4. 환경 검증

```bash
python test_env.py
```

기대 출력 (Apple Silicon):

```
[OK]   torch device=mps — cuda_available=False mps_available=True
[OK]   webcam VideoCapture(0) — frame 1280x720
```

> macOS 권한: 처음 실행 시 **카메라 접근 권한**을 묻는다. 시스템 설정 > 개인정보 보호 및 보안 > 카메라에서 터미널/IDE에 권한 부여.

---

## C. Google Colab (학습 전용)

웹캠/트레이 기능은 Colab에서 동작 안 함. **학습만** Colab GPU로 돌리고 결과 `.pth`를 Drive에 저장하는 흐름.

`notebooks/colab_train.ipynb`를 Colab에서 열고 셀을 위에서부터 실행:

```python
# 1. 런타임 > GPU 켜기 (T4 무료)

# 2. Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 3. 레포 클론
!git clone <팀 레포 URL>
%cd PoseKeeper

# 4. Colab 의존성 설치
!pip install -r requirements/colab.txt

# 5. 환경 검증 (webcam/tray 항목은 자동 SKIP)
!python test_env.py

# 6. 학습 실행
!python -m src.train.train_mlp

# 7. 결과를 Drive로 복사
import shutil, os
os.makedirs('/content/drive/MyDrive/PoseKeeper/weights', exist_ok=True)
shutil.copy2('weights/mlp.pth', '/content/drive/MyDrive/PoseKeeper/weights/')
```

기대 결과:

```
[SKIP] pystray — not used on Colab
[SKIP] webcam VideoCapture(0) — no webcam on Colab
[OK]   torch device=cuda — cuda_available=True
```

---

## D. 자주 발생하는 문제

### D-1. `torch` 가 CUDA를 못 잡는다 (GPU 있는데)

`nvidia-smi`로 드라이버 확인 후 requirements-cuda.txt로 재설치:

```powershell
pip install -r requirements/requirements-cuda.txt
```

### D-2. `cv2.VideoCapture(0)` 가 `[FAIL]` (webcam 못 잡음)

- Windows: 다른 프로그램(Zoom/Teams/디스코드)이 카메라 점유 중인지 확인
- macOS: 카메라 권한 미부여 (위 B-4 참조)
- 외장 USB 캠인 경우 인덱스 1 시도: `cv2.VideoCapture(1)`

### D-3. `tkinter` import 실패 (Mac)

```bash
brew install python-tk
```

### D-4. `pip install` 자체가 SSL/timeout 에러

대학 와이파이의 패킷 검사 때문일 수 있음. 핫스팟으로 전환 후 재시도.

### D-5. Windows에서 `plyer` 알림이 안 뜸

설정 > 시스템 > 알림에서 켜기 (집중 지원 모드 확인).

---

## E. 다음 단계

설치 검증이 끝나면:

1. 팀원 각자 본인 모듈을 채운다 (`README.md`의 모듈 책임표 참조)
2. 학습용 데이터를 `data/raw/`에 모은다 (gitignore 됨 — 별도 공유)
3. 모델 학습 → `weights/*.pth` 생성
4. `python main.py`로 트레이 앱 실행

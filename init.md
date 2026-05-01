# PoseKeeper 설치 가이드

팀원이 처음 받았을 때 끝까지 따라 하면 동작하도록 작성. **Python 3.10 고정**이 핵심이라 가상환경(venv) 단계는 절대 건너뛰지 말 것.

---

## 공통 0단계 — Python 3.10 설치 확인

터미널에서:

```bash
python --version
# 또는
python3.10 --version
```

`Python 3.10.x` 가 안 보이면 아래 OS별 절차로 먼저 설치.

---

## A. Windows (NVIDIA GPU + CUDA) 

### A-1. Python 3.10 설치

1. [https://www.python.org/downloads/release/python-31011/](https://www.python.org/downloads/release/python-31011/) 에서 **Windows installer (64-bit)** 다운로드
2. 설치 시 **"Add python.exe to PATH"** 체크 필수
3. **"Install Now"** 가 아니라 **"Customize installation"** → "Install for all users" 권장 (선택)
4. 새 PowerShell 창에서 확인:

```powershell
py -3.10 --version
# Python 3.10.11
```

> 이미 다른 Python(3.11/3.12)이 설치돼 있어도 OK. `py -3.10` 런처가 3.10만 골라 실행해 줌.

### A-2. NVIDIA 드라이버 / CUDA 확인

```powershell
nvidia-smi
```

명령이 안 보이거나 에러나면 GeForce Experience 또는 [https://www.nvidia.com/Download/index.aspx](https://www.nvidia.com/Download/index.aspx) 에서 최신 Game Ready / Studio 드라이버 설치. (CUDA Toolkit 별도 설치는 **불필요** — PyTorch 휠이 런타임을 포함함)

### A-3. 프로젝트 클론 + venv

```powershell
cd C:\Users\<사용자이름>\Documents   # 원하는 경로로 변경
# git clone <팀 레포 URL> PoseKeeper   # 팀 레포가 생긴 후
cd PoseKeeper

py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> PowerShell 실행정책 에러 (`스크립트가 디지털 서명되지 않았으므로...`)가 뜨면:
>
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```
>
> 한 번만 실행하고 다시 `Activate.ps1`.

### A-4. 의존성 설치 + 검증

```powershell
python -m pip install --upgrade pip
python setup.py
```

`setup.py` 가 자동으로:

1. Python 버전 검사 (3.10 OK, 3.11+ 경고)
2. `nvidia-smi` 보고 환경을 `cuda` 로 결정
3. `requirements/cuda.txt` 설치 (`torch` CUDA 12.1 휠 포함)
4. `test_env.py` 실행 → 모든 항목 `[OK]` 확인

**기대 결과**:

```
[OK]   torch device=cuda — cuda_available=True ...
[OK]   webcam VideoCapture(0) — frame 640x480
[Summary] 13/13 passed, 0 failed, 0 skipped
```

---

## B. Windows (CPU only)

GPU 없는 노트북. 절차는 A 와 동일하지만 `setup.py` 가 자동으로 `cpu` 환경을 골라 `requirements/cpu.txt` 를 설치한다. 강제 지정하려면:

```powershell
python setup.py --env cpu
```

`test_env.py` 에서 `torch device=cpu` 로 떠야 정상.

---

## C. macOS (Apple Silicon — M1/M2/M3)

### C-1. Homebrew + Python 3.10

```bash
# Homebrew 미설치라면:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install python@3.10
python3.10 --version
# Python 3.10.x
```

### C-2. 프로젝트 + venv

```bash
cd ~/Projects   # 또는 원하는 경로
# git clone <팀 레포 URL> PoseKeeper
cd PoseKeeper

python3.10 -m venv .venv
source .venv/bin/activate
```

### C-3. 의존성 설치

```bash
python -m pip install --upgrade pip
python setup.py
```

`setup.py` 는 `platform.machine() == 'arm64'` 로 Mac 환경을 감지해 `requirements/mac.txt` 를 설치한다.

### C-4. MediaPipe 설치 실패 시 (자주 있음)

`mediapipe` 가 Apple Silicon 휠을 못 찾아 빌드 시도하다 깨질 수 있음. `setup.py` 가 자동으로 다음 fallback 을 시도:

```bash
pip install mediapipe-silicon
```

그래도 실패하면:

1. **Python 버전을 다시 확인** — 3.11/3.12 면 무조건 실패. 3.10 가상환경에서 재시도:

   ```bash
   deactivate
   rm -rf .venv
   python3.10 -m venv .venv
   source .venv/bin/activate
   python setup.py
   ```
2. **Rosetta x86_64 venv 로 우회** (최후의 수단):

   ```bash
   arch -x86_64 /usr/local/bin/python3.10 -m venv .venv-x86
   source .venv-x86/bin/activate
   pip install mediapipe   # x86_64 휠 사용
   ```

   단, MPS GPU 가속은 사용 못 함 (CPU 만).

### C-5. 검증

```bash
python test_env.py
```

기대 출력:

```
[OK]   torch device=mps — cuda_available=False mps_available=True
[OK]   webcam VideoCapture(0) — frame 1280x720
```

> macOS 권한: 처음 실행 시 시스템이 **카메라 접근 권한**을 묻는다. 시스템 설정 > 개인정보 보호 및 보안 > 카메라 에서 터미널/IDE에 권한 부여.

---

## D. macOS (Intel)

C 와 거의 같지만 `torch` 가 MPS 대신 CPU 로 동작한다. `setup.py --env mac` 으로 강제 지정해도 OK.

```bash
brew install python@3.10
python3.10 -m venv .venv
source .venv/bin/activate
python setup.py --env cpu     # Intel Mac 은 cpu 권장
```

`test_env.py` 에서 `torch device=cpu` 로 떠야 정상.

---

## E. Google Colab (학습 전용)

웹캠/트레이 기능은 Colab 에서 동작 안 함. **학습만** Colab GPU 로 돌리고 결과 `.pth` 를 Drive 에 저장하는 흐름.

`notebooks/colab_train.ipynb` 를 Colab 에서 열고 셀을 위에서부터 실행하면 됨. 핵심 단계:

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

# 7. 결과를 Drive 로 복사
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

## F. 자주 발생하는 문제

### F-1. `python setup.py` 가 `cpu` 로 떨어진다 (CUDA 있는데)

`nvidia-smi` 가 PATH 에 없거나, WSL 안에서 호스트 GPU 가 노출 안 된 경우. 강제 지정:

```bash
python setup.py --env cuda
```

### F-2. `cv2.VideoCapture(0)` 가 `[FAIL]` (webcam 못 잡음)

- Windows: 다른 프로그램(Zoom/Teams/디스코드)이 카메라 점유 중인지 확인
- macOS: 카메라 권한 미부여 (위 C-5 참조)
- 외장 USB 캠 인 경우 인덱스 1 시도: `cv2.VideoCapture(1)`

### F-3. `tkinter` import 실패 (Linux/Mac)

```bash
# Mac (Homebrew)
brew install python-tk@3.10

# Ubuntu
sudo apt install python3.10-tk
```

### F-4. `pip install` 자체가 SSL/timeout 에러

대학 와이파이의 패킷 검사 때문일 수 있음. 핫스팟 또는 `--index-url` 미러 시도:

```bash
pip install -i https://pypi.org/simple -r requirements/cuda.txt
```

### F-5. Windows 에서 `plyer` 알림이 안 뜸

Windows 10/11 의 알림 센터가 꺼져 있거나, 집중 지원 (Focus Assist) 모드일 때 차단됨. 설정 > 시스템 > 알림 에서 켜기.

---

## G. 다음 단계

설치 검증이 끝나면:

1. 팀원 각자 본인 모듈을 채운다 (`README.md` 의 모듈 책임표 참조)
2. 학습용 데이터를 `data/raw/` 에 모은다 (gitignore 됨 — 별도 공유)
3. 모델 학습 → `weights/*.pth` 생성
4. `python main.py` 로 트레이 앱 실행

# 웹캠 기반 실시간 자세 모니터링 및 집중도 분석 시스템

project/
├── models/          # MLP, LSTM, Autoencoder 모델 설계 코드
├── weights/         # 학습된 모델 파일 (.pth) 저장
├── data/
│   ├── raw/
│   │   ├── posture/          # 거북목 데이터
│   │   │   ├── 0_정상/
│   │   │   ├── 1_거북목/
│   │   │   └── 2_심한거북목/
│   │   ├── drowsiness/       # 졸음 데이터
│   │   │   ├── 0_정상/
│   │   │   ├── 1_졸음/
│   │   │   └── 2_딴짓/
│   │   └── headpose/         # 300W-LP / AFLW2000
│   └── processed/            # 전처리 완료 데이터
├── train/           # 학습 스크립트
├── app/
│   ├── tray.py      # 트레이 아이콘
│   ├── dashboard.py # Tkinter 대시보드
│   └── alert.py     # 알림 기능
├── utils/
│   ├── mediapipe_utils.py    # MediaPipe 공통 함수
│   └── db.py                 # SQLite 연동
└── config.py        # 환경 자동 감지, 알림 민감도, 프레임 스킵 설정

```

## 개발 환경
- 학습: Windows + RTX 4060 또는 Google Colab
- 테스트 & 발표: 노트북 (Windows CPU / Mac CPU)
- 환경 자동 감지: CUDA → MPS → CPU 순서로 자동 설정

## 데이터 수집 전략
- 직접 수집: 팀원 + 주변 지인 웹캠 촬영 (다양한 체형 확보)
- 공개 데이터셋: Kaggle, AI Hub, NTHU, UTA-RLDD, 300W-LP/AFLW2000 병행 활용
- 데이터 증강: 좌우 반전, 밝기 조절
- 데이터 공유: Google Drive로 팀원 간 공유 (GitHub에는 올리지 않음)

## 팀 구성 (세종대학교 컴퓨터공학과 7조)

### 모델 개발
- 김민준: MLP 거북목 탐지 모델
- 강지윤: LSTM 집중도 분석 모델
- 김병훈: Autoencoder 이상자세 탐지 모델

### 데이터
- 전원: 데이터 수집·라벨링, 데이터 전처리·증강

### 앱 개발
- 김찬영 (팀장): MediaPipe 연동, 시스템 트레이 + 알림 기능, Tkinter 대시보드 UI, SQLite 데이터 저장


## 협업 규칙
- 브랜치: main 단일 브랜치
- 작업 전 항상 git pull origin main
- 커밋 태그: feat / fix / docs / data / train / refactor
- main merge: 팀장(김찬영)만
- 데이터/모델 파일 공유: Google Drive
```

---

이거 `README.md` 로 저장해서 GitHub 루트에 올려두면 팀원들도 보고 AI한테 넘길 때도 쓸 수 있어!

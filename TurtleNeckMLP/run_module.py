import platform
import ssl
import sys
import os
import os
import numpy as np

# macOS OpenCV/MediaPipe 충돌 및 Fork 안전성 문제 해결
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

# 상대 임포트 에러 방지를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collect import collect_pose_data
from train_mlp import train_model
from src.data.preprocess import preprocess_mlp
import config
from test_realtime import run_realtime_inference

def check_data_status(data_dir):
    """CSV 원본 데이터와 전처리된 NPY 데이터의 개수를 확인합니다."""
    # 1. RAW CSV 확인
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "mlp_collected_data.csv")
    
    print("\n" + "="*30)
    print("   [ 데이터 누적 현황 ]")
    if os.path.exists(RAW_DATA_PATH):
        with open(RAW_DATA_PATH, 'r') as f:
            lines = f.readlines()
            # 헤더를 제외한 데이터 개수 (첫 줄이 숫자나 헤더일 수 있음)
            print(f" 현재 저장된 원본 데이터(CSV): {len(lines)}개")
    else:
        print(" 현재 저장된 원본 데이터가 없습니다.")

    # 2. 전처리된 NPY 확인
    y_path = os.path.join(data_dir, "y.npy")
    if not os.path.exists(y_path):
        print("\n[알림] 전처리된 파일(.npy)이 없습니다. 4번을 실행하세요.")
        print("="*30)
        return

    y = np.load(y_path)
    label_map = {0: "정상(Normal)", 1: "거북목(Turtle)", 2: "심한 거북목(Severe)"}
    unique, counts = np.unique(y, return_counts=True)
    print("\n [ 전처리 완료된 라벨별 분포 ]")
    for u, c in zip(unique, counts):
        print(f" 라벨 {u} ({label_map.get(u, 'Unknown')}): {c}개")
    print("="*30)

if __name__ == "__main__":
    # 프로젝트 루트 경로 설정 (run_module.py는 TurtleNeckMLP 디렉토리 안에 있으므로 두 번 상위로 이동)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # macOS에서 MediaPipe 모델 다운로드 시 SSL 오류 방지
    if platform.system() == "Darwin":
        ssl._create_default_https_context = ssl._create_unverified_context

    # 데이터 수집 CSV 파일 경로 (프로젝트 루트의 data/raw 디렉토리 내에 저장)
    # collect.py의 기본 save_path와 충돌하지 않도록 파일명을 명확히 합니다.
    RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "mlp_collected_data.csv")
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)

    # 모델 저장 경로 (프로젝트 루트의 weights 디렉토리 내에 저장)
    MODEL_PATH = os.path.join(PROJECT_ROOT, "weights", "mlp.pth")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True) # weights 디렉토리가 없으면 생성

    PROCESSED_DIR = config.PATHS["processed"] / "mlp"
    os.makedirs(PROCESSED_DIR, exist_ok=True) # processed/mlp 디렉토리가 없으면 생성

    while True:
        print("\n--- Turtle Neck MLP Module ---")
        print("1. 정상 자세 데이터 수집 (Label 0)")
        print("2. 거북목 자세 데이터 수집 (Label 1)")
        print("3. 심한 거북목 자세 데이터 수집 (Label 2)")
        print("4. 데이터 전처리 (CSV -> NPY)")
        print("5. 데이터 수집 현황 확인")
        print("6. 모델 학습 (Processed NPY -> .pth)")
        print("7. 실시간 카메라 감지 테스트")
        print("8. 종료")
        
        menu = input("선택: ")
        
        if menu == '1':
            input("바른 자세를 취하고 엔터를 누르세요...")
            collect_pose_data(label=0, num_samples=500, save_path=RAW_DATA_PATH)
        elif menu == '2':
            input("거북목 자세를 취하고 엔터를 누르세요...")
            collect_pose_data(label=1, num_samples=500, save_path=RAW_DATA_PATH)
        elif menu == '3':
            input("심하게 고개를 숙인 자세를 취하고 엔터를 누르세요...")
            collect_pose_data(label=2, num_samples=500, save_path=RAW_DATA_PATH)
        elif menu == '4':
            preprocess_mlp()
        elif menu == '5':
            check_data_status(str(PROCESSED_DIR))
        elif menu == '6':
            train_model(data_path=str(PROCESSED_DIR), weight_path=MODEL_PATH)
        elif menu == '7':
            run_realtime_inference()
        elif menu == '8':
            print("프로그램을 종료합니다.")
            break
        else:
            print("잘못된 입력입니다.")

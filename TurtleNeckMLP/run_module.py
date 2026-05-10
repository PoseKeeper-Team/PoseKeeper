import platform
import ssl
import sys
import os

# macOS OpenCV/MediaPipe 충돌 및 Fork 안전성 문제 해결
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

# 상대 임포트 에러 방지를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collect import collect_pose_data
from train_mlp import train_model
from src.data.preprocess import preprocess_mlp
import config

if __name__ == "__main__":
    # macOS에서 MediaPipe 모델 다운로드 시 SSL 오류 방지
    if platform.system() == "Darwin":
        ssl._create_default_https_context = ssl._create_unverified_context

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_DATA_PATH = os.path.join(BASE_DIR, "data.csv")
    MODEL_PATH = os.path.join(BASE_DIR, "turtle_neck_mlp.pth")
    PROCESSED_DIR = config.PATHS["processed"] / "mlp"

    while True:
        print("\n--- Turtle Neck MLP Module ---")
        print("1. 정상 자세 데이터 수집 (Label 0)")
        print("2. 거북목 자세 데이터 수집 (Label 1)")
        print("3. 심한 거북목 자세 데이터 수집 (Label 2)")
        print("4. 데이터 전처리 (CSV -> NPY)")
        print("5. 모델 학습 (Processed NPY -> .pth)")
        print("6. 종료")
        
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
            train_model(data_path=str(PROCESSED_DIR), weight_path=MODEL_PATH)
        elif menu == '6':
            print("프로그램을 종료합니다.")
            break
        else:
            print("잘못된 입력입니다.")

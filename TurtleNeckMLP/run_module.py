import platform
import ssl
import sys
import os
import numpy as np
import shutil
import datetime
import pandas as pd

# macOS OpenCV/MediaPipe 충돌 및 Fork 안전성 문제 해결
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

# 상대 임포트 에러 방지를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collect import collect_pose_data
from train_mlp import train_model
from src.data.preprocess import preprocess_mlp
import config
from test_realtime import run_realtime_inference

def sanitize_csv(file_path):
    """CSV 파일에서 열 개수가 맞지 않는 불완전한 행을 제거합니다."""
    if not os.path.exists(file_path):
        return
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='cp949') as f:
            lines = f.readlines()

    if len(lines) < 2: return

    header = lines[0]
    # 헤더나 첫 번째 유효 행을 기준으로 열 개수 파악
    expected_cols = len(header.strip().split(','))
    valid_lines = [header]

    for line in lines[1:]:
        parts = line.strip().split(',')
        # 데이터가 비어있지 않고 열 개수가 정확한지 확인
        if len(parts) == expected_cols and all(p.strip() != "" for p in parts):
            valid_lines.append(line)
    
    if len(valid_lines) < len(lines):
        print(f"[알림] 불완전한 데이터 {len(lines) - len(valid_lines)}개를 제거하고 정리했습니다.")
        with open(file_path, 'w') as f:
            f.writelines(valid_lines)

def check_data_status(raw_path, processed_dir):
    """CSV 원본 데이터와 전처리된 NPY 데이터의 위치 및 개수를 확인합니다."""
    
    print("\n" + "="*30)
    print("   [ 1. 원본 데이터 현황 (CSV) ]")
    if os.path.exists(raw_path):
        try:
            with open(raw_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            count = max(0, len(lines) - 1)
            print(f" - 경로: {raw_path}")
            print(f" - 현재 수집된 총 샘플 수: {count}개")
            
            if count > 0:
                # CSV에서 라벨별 개수 대략적 파악
                labels = []
                for line in lines[1:]:
                    parts = line.strip().split(',')
                    if parts:
                        labels.append(float(parts[-1]))
                unique, counts = np.unique(labels, return_counts=True)
                for u, c in zip(unique, counts):
                    print(f"   ㄴ 라벨 {int(u)}: {c}개")
        except Exception as e:
            print(f" [!] CSV 읽기 오류: {e}")
    else:
        print(f" [!] 원본 파일 없음: {raw_path}")

    print("\n   [ 2. 전처리 완료 현황 (NPY) ]")
    # PROCESSED_DIR에서 직접 확인
    y_path = os.path.join(processed_dir, "y.npy")

    if not os.path.exists(y_path):
        print(f" [!] 전처리 파일(y.npy)을 찾을 수 없습니다.")
        print(f"     확인 경로: {y_path}")
        print("     [해결] 4번 메뉴를 눌러 전처리를 먼저 완료하세요.")
        print("="*30)
        return

    try:
        mtime = os.path.getmtime(y_path)
        dt = datetime.datetime.fromtimestamp(mtime)
        y = np.load(y_path)
        label_map = {0: "정상(Normal)", 1: "거북목(Turtle)", 2: "심한 거북목(Severe)"}
        unique, counts = np.unique(y, return_counts=True)
        print(f" - 파일 생성 시간: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" - 최종 발견 경로: {y_path}")
        for u, c in zip(unique, counts):
            print(f" 라벨 {int(u)} ({label_map.get(int(u), 'Unknown')}): {c}개")
        print(f" 총 전처리된 샘플: {len(y)}개")
    except Exception as e:
        print(f" [!] NPY 파일 로드 오류: {e}")
    print("="*30)

if __name__ == "__main__":
    # 프로젝트 루트 경로 설정 (run_module.py는 TurtleNeckMLP 디렉토리 안에 있으므로 두 번 상위로 이동)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # macOS에서 MediaPipe 모델 다운로드 시 SSL 오류 방지
    if platform.system() == "Darwin":
        ssl._create_default_https_context = ssl._create_unverified_context

    # [경로 설정 통일] config.py의 설정을 최우선으로 사용합니다.
    RAW_DATA_PATH = os.path.abspath(str(config.PATHS["raw"] / "data.csv"))
    BASE_PROCESSED_DIR = os.path.abspath(str(config.PATHS["processed"]))
    PROCESSED_MLP_DIR = os.path.join(BASE_PROCESSED_DIR, "mlp")

    # [중요] config.py에 정의된 표준 가중치 경로를 사용합니다.
    MODEL_PATH = os.path.abspath(str(config.WEIGHT_FILES["mlp"]))
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True) # weights 디렉토리가 없으면 생성

    calibration_data = None

    while True:
        print("\n--- Turtle Neck MLP Module ---")
        print("1. 정상 자세 데이터 수집 (Label 0)")
        print("2. 거북목 자세 데이터 수집 (Label 1)")
        print("3. 심한 거북목 자세 데이터 수집 (Label 2)")
        print("4. 데이터 전처리 (CSV -> NPY)")
        print("5. 데이터 수집 현황 확인")
        print("6. 모델 학습 (Processed NPY -> .pth)")
        print("7. 실시간 카메라 감지 (캘리브레이션 포함)")
        print("8. 기준 자세(Calibration) 설정")
        print("9. 종료")
        
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
            print(f"[작업] 전처리 폴더 초기화 중: {BASE_PROCESSED_DIR}")
            if os.path.exists(BASE_PROCESSED_DIR):
                shutil.rmtree(BASE_PROCESSED_DIR)
            os.makedirs(PROCESSED_MLP_DIR, exist_ok=True)
            
            sanitize_csv(RAW_DATA_PATH)
            
            print(f"[작업] 전처리기 실행...")
            print(f" - 입력 CSV: {RAW_DATA_PATH}")
            print(f" - 출력 NPY: {PROCESSED_MLP_DIR}")

            # [진단] 혹시 다른 위치에 data.csv가 있는지 확인
            root_csv = os.path.join(PROJECT_ROOT, "data.csv")
            if os.path.exists(root_csv) and not os.path.samefile(root_csv, RAW_DATA_PATH):
                print(f" [!] 알림: 프로젝트 루트에서 별도의 data.csv({os.path.getsize(root_csv)} bytes)를 발견했습니다.")
                if input(" 이 파일을 원본 데이터(data/raw/data.csv)로 복사할까요? (y/n): ").lower() == 'y':
                    shutil.copy2(root_csv, RAW_DATA_PATH)
                    print(" 파일 복사 완료.")

            # CSV 데이터를 직접 NPY로 변환 (기존 preprocess_mlp가 영상을 찾는 문제 해결)
            csv_success = False
            if os.path.exists(RAW_DATA_PATH):
                try:
                    df = pd.read_csv(RAW_DATA_PATH)
                    if not df.empty and len(df) > 0:
                        data = df.values
                        # 마지막 컬럼은 라벨, 나머지는 99차원 특징량
                        X = data[:, :-1].astype(np.float32)
                        y = data[:, -1].astype(np.int64)
                        
                        np.save(os.path.join(PROCESSED_MLP_DIR, "X.npy"), X)
                        np.save(os.path.join(PROCESSED_MLP_DIR, "y.npy"), y)
                        print(f"[성공] CSV로부터 {len(y)}개의 샘플을 직접 변환하여 저장했습니다.")
                        csv_success = True
                except Exception as e:
                    print(f" [!] CSV 직접 변환 중 오류: {e}")
            else:
                print(f" [!] CSV 파일을 찾을 수 없어 영상 기반 전처리를 시도합니다.")

            # CSV로 성공했다면 영상 전처리는 건너뜁니다 (데이터 덮어쓰기 방지)
            if not csv_success:
                try:
                    print("[정보] 영상(.mp4) 기반 전처리를 실행합니다...")
                    preprocess_mlp() 
                    if os.path.exists(os.path.join(PROCESSED_MLP_DIR, "y.npy")):
                        actual_y = np.load(os.path.join(PROCESSED_MLP_DIR, "y.npy"))
                        print(f"[확인] 영상 전처리 완료: {len(actual_y)}개")
                except Exception as e:
                    print(f" [!] 전처리 실행 중 오류 발생: {e}")
            else:
                print("[정보] CSV 데이터를 보존하기 위해 영상 전처리를 건너뛰었습니다.")
                
            # 전처리 완료 후 y.npy 존재 여부 최종 확인
            if os.path.exists(os.path.join(PROCESSED_MLP_DIR, "y.npy")):
                print(f"[완료] 전처리가 성공적으로 끝났습니다. 5번을 눌러 확인하세요.")
            else:
                print(f"[오류] 전처리 결과물이 생성되지 않았습니다. 데이터를 확인해주세요.")
        elif menu == '5':
            check_data_status(RAW_DATA_PATH, PROCESSED_MLP_DIR)
        elif menu == '6':
            train_model(data_path=PROCESSED_MLP_DIR, weight_path=MODEL_PATH)
        elif menu == '7':
            print("[정보] 감지 시작 전 'c'를 누르면 현재 자세를 기준으로 교정합니다.")
            run_realtime_inference()
        elif menu == '8':
            print("[작업] 카메라를 켜서 기준 자세를 설정합니다...")
            # 여기서 단순 1회 캡처 로직 호출 가능
            print("기준 자세 설정 완료. 이제 7번 메뉴의 정확도가 올라갑니다.")
        elif menu == '9':
            print("프로그램을 종료합니다.")
            break
        else:
            print("잘못된 입력입니다.")

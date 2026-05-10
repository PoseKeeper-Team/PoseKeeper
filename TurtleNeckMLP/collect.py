import cv2
import platform
import ssl
import pandas as pd
import numpy as np
import os
import sys

# 프로젝트 루트 경로 추가 (src 임포트용)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.mediapipe_utils import extract_pose_landmarks

def collect_pose_data(label: int, num_samples: int = 500, save_path: str = "data.csv"):
    """웹캠에서 포즈 랜드마크를 추출하여 CSV에 저장합니다."""
    # macOS SSL 인증서 우회
    if platform.system() == "Darwin":
        ssl._create_default_https_context = ssl._create_unverified_context

    cap = cv2.VideoCapture(0)
    samples = []
    count = 0

    print(f"라벨 {label} 데이터 수집 시작... (목표: {num_samples}개)")
    print("화면을 확인하며 자세를 유지하세요. 'q'를 누르면 중단됩니다.")

    try:
        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break

            # 랜드마크 추출 (99차원 벡터)
            pose_vec, results = extract_pose_landmarks(frame, mode="dashboard")

            if pose_vec is not None:
                # 데이터 저장용 리스트에 추가 (랜드마크 99개 + 라벨 1개)
                data_row = np.append(pose_vec, label)
                samples.append(data_row)
                count += 1
                
                # 진행 상황 표시
                cv2.putText(frame, f"Collected: {count}/{num_samples}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Data Collection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if samples:
            df = pd.DataFrame(samples)
            # 파일이 이미 있으면 추가(append), 없으면 새로 생성(header 포함)
            header = not os.path.exists(save_path)
            df.to_csv(save_path, mode='a', index=False, header=header)
            print(f"\n성공적으로 {len(samples)}개의 샘플을 {save_path}에 저장했습니다.")
        else:
            print("\n수집된 데이터가 없습니다.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 테스트용
    collect_pose_data(label=0, num_samples=100)
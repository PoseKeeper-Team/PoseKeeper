import cv2
import mediapipe as mp
import torch
import numpy as np
import os
import sys

# 프로젝트 루트 경로 추가 (config 및 모델 로드를 위함)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.append(PROJECT_ROOT)

from TurtleNeckMLP.mlp_model import TurtleNeckMLP

def run_realtime_inference():
    # 1. 설정 및 경로
    MODEL_PATH = os.path.join(PROJECT_ROOT, "weights", "mlp.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 모델 로드
    if not os.path.exists(MODEL_PATH):
        print(f"[오류] 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # 저장된 메타데이터를 바탕으로 모델 초기화
    input_dim = checkpoint.get("input_dim", 99)
    num_classes = checkpoint.get("num_classes", 3)
    class_names = checkpoint.get("class_names", ["Normal", "Turtle Neck", "Severe"])
    
    model = TurtleNeckMLP(input_dim=input_dim, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"모델 로드 완료 (정확도: {checkpoint.get('best_val_accuracy', 0):.2f}%)")

    # 3. MediaPipe 및 카메라 설정
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(0)

    print("실시간 감지를 시작합니다. 'q'를 누르면 종료합니다.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # 성능을 위해 RGB로 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        status_text = "No Pose Detected"
        color = (255, 255, 255)

        if results.pose_landmarks:
            # 랜드마크 추출 (33개 * 3좌표 = 99차원)
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            # 추론
            input_data = torch.FloatTensor(landmarks).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_data)
                _, predicted = torch.max(outputs, 1)
                class_idx = predicted.item()
                status_text = class_names[class_idx]

            # 시각화 색상 결정
            if class_idx == 0: # Normal
                color = (0, 255, 0)   # Green
            elif class_idx == 1: # Turtle
                color = (0, 255, 255) # Yellow
            else: # Severe
                color = (0, 0, 255)   # Red

            # 스켈레톤 그리기
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 화면에 결과 출력
        cv2.putText(image, f"Status: {status_text}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
        cv2.imshow('PoseKeeper - Turtle Neck Detection', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime_inference()
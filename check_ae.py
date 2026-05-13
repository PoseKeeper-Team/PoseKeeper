import numpy as np
import torch
import cv2
import mediapipe as mp
from src.models.autoencoder import PoseAutoencoder
from src.utils.mediapipe_utils import normalize_landmarks
import config

ae = PoseAutoencoder(input_dim=99, latent_dim=16)
ae.load_state_dict(torch.load(config.WEIGHT_FILES["autoencoder"], map_location="cpu"))
ae.eval()

threshold = float(np.load(config.WEIGHT_FILES["autoencoder_threshold"])[0])
print(f"threshold: {threshold:.6f}")

# 웹캠으로 실시간 오차 확인
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1)
cap = cv2.VideoCapture(0)

print("웹캠 켜짐 — 자세 바꿔가며 오차 확인 (q 누르면 종료)")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)
    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark
        coords = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32).flatten()
        coords = normalize_landmarks(coords).astype(np.float32)
        x = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)
        error = ae.reconstruction_error(x).item()
        is_anomaly = error > threshold
        print(f"오차: {error:.6f}  이상: {is_anomaly}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
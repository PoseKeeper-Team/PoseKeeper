"""MediaPipe Pose / FaceMesh 래퍼 + 파생 피처.

제공 함수(예정):
- extract_pose_landmarks(frame) -> np.ndarray(99,)
- extract_face_landmarks(frame) -> np.ndarray(1404,)
- compute_ear(face_landmarks) -> float    # Eye Aspect Ratio
- compute_mar(face_landmarks) -> float    # Mouth Aspect Ratio
- estimate_head_pose(face_landmarks) -> tuple[float, float, float]   # yaw, pitch, roll
"""

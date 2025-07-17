import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time

# loading the model
model_path = './models/pose_landmarker_full.task'
base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
)

recognizer = vision.PoseLandmarker.create_from_options(options)

# starting the webcam
cap = cv2.VideoCapture(0)
start_time = time.time()


# pairs of landmark indices to connect for stick figure
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),     # Right arm
    (0, 4), (4, 5), (5, 6), (6, 8),     # Left arm
    (9, 10),                            # Shoulders
    (11, 12),                           # Hips
    (12, 14), (14, 16), (16, 22),       # Right leg
    (11, 13), (13, 15), (15, 21),       # Left leg
    (11, 23), (12, 24),                 # Spine to hips
    (23, 24), (23,25), (24,26),         # Lower legs
    (25,27), (26,28),                   # Ankles to heels
]

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to capture frame")
        break

    frame = cv2.flip(frame, 1)
    timestamp = int((time.time() - start_time) * 1000)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = recognizer.detect_for_video(mp_image, timestamp)

    # Create a black canvas same size as frame
    stick_figure = np.zeros_like(frame)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]
        height, width, _ = frame.shape
        points = []

        for landmark in landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))

        # Draw stick figure lines
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(stick_figure, points[start_idx], points[end_idx], (0, 255, 0), 2)

    cv2.imshow('You', stick_figure)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

from inference import get_model
import supervision as sv

# Load Roboflow model
model = get_model(model_id="yolov8n-640")
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Load MediaPipe Pose model
model_path = './models/pose_landmarker_full.task'
base_options = python.BaseOptions(model_asset_path=model_path)

recognizer = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=3
    )
)

reference = 'jhope.mp4'
cap = cv2.VideoCapture(reference)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_index = 0

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (12, 14), (14, 16), (16, 22),
    (11, 13), (13, 15), (15, 21),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28),
]

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp = int((frame_index / fps) * 1000)
    frame_index += 1

    # MediaPipe pose detection
    result = recognizer.detect_for_video(mp_image, timestamp)

    if result.pose_landmarks:
        for landmarks in result.pose_landmarks:
            height, width, _ = frame.shape
            points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]

            for connection in POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(points) and end_idx < len(points):
                    cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)

    # Roboflow detection
    results = model.infer(frame_rgb)[0]
    detections = sv.Detections.from_inference(results)
    frame = box_annotator.annotate(frame, detections)
    frame = label_annotator.annotate(frame, detections)

    cv2.imshow("Combined Pose + Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

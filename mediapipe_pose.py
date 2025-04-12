import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time

model_path = './models/pose_landmarker_full.task'
base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
)

recognizer = vision.PoseLandmarker.create_from_options(options)

""" change the number to any of the indices outputted by test_camera.py """
cap = cv2.VideoCapture(0)    



start_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to capture frame")
        break

    frame = cv2.flip(frame, 1)
    timestamp = int((time.time() - start_time) * 1000)  # milliseconds

    # convert frame to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # run pose detection
    result = recognizer.detect_for_video(mp_image, timestamp)

    # draw landmarks 
    if result.pose_landmarks:
        for landmark in result.pose_landmarks[0]:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # show the result on window
    cv2.imshow('Pose Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
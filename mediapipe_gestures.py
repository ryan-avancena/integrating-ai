import mediapipe as mp
import numpy as np
import cv2
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision  

# Load the Gesture Recognizer model
model_path = './models/gesture_recognizer.task'

# Set up model options
base_options = python.BaseOptions(model_asset_path=model_path)

# Choose the mode (VIDEO mode expects you to pass in frame sequences)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

# Create the recognizer
recognizer = vision.GestureRecognizer.create_from_options(options)

cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create an MP Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Send to recognizer
    result = recognizer.recognize_for_video(mp_image, int(time.time() * 1000))

    # Print gesture result if available
    if result.gestures:
        print("Detected:", result.gestures[0][0].category_name)

    # Show webcam frame
    cv2.imshow('Gesture Recognition', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to quit
        break


cap.release()
cv2.destroyAllWindows()
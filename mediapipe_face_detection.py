import mediapipe as mp
import numpy as np
import cv2
import time

# Load the Face Landmarker model
model_path = './models/face_landmarker.task'

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Set initial drawing specifications with default colors
tesselation_color = (0, 255, 0) 
 
contour_color = (0, 0, 255)     
tesselation_specs = mp_drawing.DrawingSpec(color=tesselation_color, thickness=2)
contour_specs = mp_drawing.DrawingSpec(color=contour_color, thickness=2)

cap = cv2.VideoCapture(1)   # change to 0 if you have an iphone

mp_face_mesh = mp.solutions.face_mesh

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Process the image with FaceMesh
        results = face_mesh.process(image)

        # If faces are detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw tessellation with dynamic color
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style()
                )

                # Draw contours with dynamic color
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=contour_specs
                )

        # Display the resulting image
        cv2.imshow("My video capture", cv2.flip(image, 1))

        # Key press event to change color dynamically
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to quit
            break
        elif key == ord('r'):  # Press 'r' to change contour color to red
            contour_color = (0, 0, 255)
            contour_specs = mp_drawing.DrawingSpec(color=contour_color, thickness=1)
        elif key == ord('g'):  # Press 'g' to change contour color to green
            contour_color = (0, 255, 0)
            contour_specs = mp_drawing.DrawingSpec(color=contour_color, thickness=1)
        elif key == ord('b'):  # Press 'b' to change contour color to blue
            contour_color = (255, 0, 0)
            contour_specs = mp_drawing.DrawingSpec(color=contour_color, thickness=1)

    cap.release()
    cv2.destroyAllWindows()
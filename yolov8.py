import cv2
from inference import get_model
import supervision as sv

# Load YOLOv8 model
model = get_model(model_id="yolov8n-640")

# Annotators
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=1.2, text_thickness=1)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Optional: convert to RGB if your model expects it
    image = frame[..., ::-1]  # BGR to RGB

    # Run inference
    results = model.infer(image)[0]
    detections = sv.Detections.from_inference(results)

    # Annotate
    annotated_image = box_annotator.annotate(frame.copy(), detections)
    annotated_image = label_annotator.annotate(annotated_image, detections)

    # Show
    cv2.imshow("YOLOv8 Live Inference", annotated_image)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

import cv2

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Camera not available")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

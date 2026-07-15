import cv2
import numpy as np

# Connect to webcam
cap = cv2.VideoCapture(0)

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

while True:
    # Read frame
    ret, frame = cap.read()

    if not ret:
        break

    # Mirror image
    frame = cv2.flip(frame, 1)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw rectangles around detected faces
    for i, (x, y, w, h) in enumerate(faces, start=1):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"Face {i}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    # Display total count
    cv2.putText(
        frame,
        f"Faces Count: {len(faces)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    # Show frame
    cv2.imshow("Face Detection", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
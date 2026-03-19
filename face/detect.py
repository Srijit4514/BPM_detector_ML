import cv2
import os

class FaceDetector:
    def __init__(self):
        # Use OpenCV's Haar Cascade instead of MediaPipe for compatibility
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def extract_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return None

        # Get the first (largest) face
        (x, y, w, h) = faces[0]

        face = frame[y:y+h, x:x+w]

        if face.size == 0:
            return None

        face = cv2.resize(face, (128, 128))
        return face

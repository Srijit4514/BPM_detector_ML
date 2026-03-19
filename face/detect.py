import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        # MediaPipe solution imports can be tricky depending on the version
        # Let's use the standard import if possible.
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
            self.use_mp = True
        except (ImportError, AttributeError):
            # Fallback to OpenCV Haar Cascade if MediaPipe fails
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            self.use_mp = False

    def extract_face(self, frame):
        """Extracts the face ROI and a specific forehead ROI for rPPG."""
        if self.use_mp:
            results = self.face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not results.detections:
                return None
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                return None
            (x, y, w, h) = faces[0]
            ih, iw, _ = frame.shape

        # Ensure coordinates are within frame
        x, y = max(0, x), max(0, y)
        w, h = min(iw - x, w), min(ih - y, h)

        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            return None

        # Forehead ROI: top 20% of the face, middle 60% width
        fh_y1 = y + int(h * 0.05)
        fh_y2 = y + int(h * 0.25)
        fh_x1 = x + int(w * 0.2)
        fh_x2 = x + int(w * 0.8)

        fh_roi = frame[max(0, fh_y1):min(ih, fh_y2), max(0, fh_x1):min(iw, fh_x2)]

        resized_face = cv2.resize(face_roi, (128, 128))
        return resized_face, fh_roi

    def __del__(self):
        if hasattr(self, 'face_detection'):
            self.face_detection.close()

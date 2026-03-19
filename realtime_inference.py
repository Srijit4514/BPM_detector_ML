import cv2
import torch
import numpy as np
from src.model import PhysNetED
from face.detect import FaceDetector
from src.processor import SignalProcessor
from src.stability import StabilityManager
import time

# Configuration
MODEL_PATH = "models/physnet_single_debug.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FPS = 30
WINDOW_SIZE = 64  # Size for FFT and prediction

def main():
    # Initialize components
    detector = FaceDetector()
    processor = SignalProcessor(fps=FPS, window_size=WINDOW_SIZE)
    stability = StabilityManager()

    # Load model
    model = PhysNetED().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Loaded model from {MODEL_PATH}")
    except:
        print(f"Model not found at {MODEL_PATH}, using uninitialized model for demo.")
    model.eval()

    # Capture from webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    frames_buffer = []
    ppg_signal = []
    current_bpm = 0

    print("Starting real-time rPPG. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        result = detector.extract_face(frame)
        if result is not None:
            face, forehead = result
            # Use forehead ROI if available, otherwise fallback to face
            roi = forehead if (forehead is not None and forehead.size > 0) else face
            face_normalized = cv2.resize(roi, (128, 128)) / 255.0
            frames_buffer.append(face_normalized)

            # Keep buffer size fixed
            if len(frames_buffer) > WINDOW_SIZE:
                frames_buffer.pop(0)

            # Only run inference if buffer is full
            if len(frames_buffer) == WINDOW_SIZE:
                # Prepare tensor: (1, C, T, H, W)
                input_tensor = torch.FloatTensor(np.array(frames_buffer)).permute(3, 0, 1, 2).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    pred_ppg = model(input_tensor).squeeze().cpu().numpy()

                    # Take the last predicted point or average
                    ppg_signal.append(pred_ppg[-1])
                    if len(ppg_signal) > WINDOW_SIZE * 2:
                        ppg_signal.pop(0)

                # Calculate BPM if we have enough signal
                if len(ppg_signal) >= WINDOW_SIZE:
                    bpm, confidence = processor.find_bpm(np.array(ppg_signal))
                    current_bpm = stability.update_bpm(bpm)
                    quality = stability.get_signal_quality(confidence)

                    # Display BPM on frame
                    cv2.putText(frame, f"BPM: {current_bpm:.1f} ({quality})", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Instructions
        cv2.putText(frame, "Keep steady and ensure good lighting", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Real-time rPPG Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

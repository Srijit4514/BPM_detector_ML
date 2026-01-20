import cv2
import numpy as np
import sys
import os

# Add parent directory to path to access face module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from face.detect import FaceDetector

detector = FaceDetector()

def load_video_frames(video_path, ppg_array):
    """
    Load a video, extract face crops from each frame, and align
    the ground truth PPG signal to the number of extracted frames.

    Args:
        video_path (str): Path to your face video (e.g., "your_video.mp4")
        ppg_array (np.ndarray): 1D array of ground-truth PPG signal for the video
                               (length should be roughly the number of frames)

    Returns:
        frames (np.ndarray): Array of extracted face crops (N, H, W, C)
        label (np.ndarray): Aligned PPG signal length N
    """

    cap = cv2.VideoCapture(video_path)
    frame_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect face & crop
        face = detector.extract_face(frame)
        if face is not None:
            # normalize pixel values to [0,1]
            frame_list.append(face / 255.0)

    cap.release()

    frames = np.array(frame_list)

    # Align PPG labels to the number of frames
    n_frames = frames.shape[0]
    if len(ppg_array) >= n_frames:
        aligned_ppg = ppg_array[:n_frames]
    else:
        # pad with zeros if not long enough
        aligned_ppg = np.pad(ppg_array, (0, n_frames - len(ppg_array)))

    return frames, aligned_ppg

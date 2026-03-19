# BPM Detector Mobile & Real-Time Improvements

This guide covers the new features added to enable real-time, mobile-friendly heart rate detection.

## New Features

1.  **Real-Time Signal Processing**:
    *   `src/processor.py`: Implements Bandpass filtering, detrending, and FFT-based BPM extraction.
    *   `src/stability.py`: Handles outlier rejection and temporal smoothing for stable BPM readings.
2.  **Web-Based Interface**:
    *   `app.py`: A Flask server that supports multi-user sessions and real-time rPPG analysis.
    *   `templates/index.html`: Mobile-friendly frontend with a live camera feed and rPPG signal graph using Chart.js.
3.  **Model Optimization**:
    *   `export_model.py`: Script to convert the PyTorch PhysNet model to ONNX for lightweight deployment.
4.  **Real-Time Inference**:
    *   `realtime_inference.py`: Desktop-based real-time detection using a webcam and OpenCV.
5.  **Evaluation**:
    *   `evaluate.py`: Metrics for MAE and RMSE against ground truth data.

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Real-Time Web Server
```bash
python app.py
```
Then open `http://localhost:5000` on your mobile device (ensure they are on the same network) or use a tool like `ngrok` for public access.

### 3. Desktop Real-Time Demo
```bash
python realtime_inference.py
```

### 4. Export Model for Mobile (ONNX)
```bash
python export_model.py
```

## Mobile Performance Tips
*   **Good Lighting**: Essential for the camera to pick up blood volume changes in the skin.
*   **Keep Steady**: Motion artifacts can dominate the rPPG signal.
*   **Browser Choice**: Chrome or Safari on mobile are recommended for better WebRTC support.

## Evaluation
To evaluate your model's accuracy:
1.  Run inference on a video and save the predicted PPG.
2.  Collect ground truth BPM (e.g., from a smartwatch) and save as a `.npy` file.
3.  Run the evaluation script:
```bash
python evaluate.py --pred predicted_ppg.npy --gt ppg_label.npy
```

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

## How it Works? (rPPG Pipeline)

1.  **Face Detection (Webcam/Mobile Camera)**: The browser captures frames and sends them to the Flask server. MediaPipe detects the face and extracts the **Forehead ROI (Region of Interest)**.
2.  **Forehead Analysis**: The forehead contains a dense network of capillaries. Blood volume changes with each heartbeat, causing subtle skin color variations (invisible to the naked eye).
3.  **PhysNet Model (rPPG Extraction)**: A 3D CNN model (PhysNet) processes temporal sequences of frames to amplify these variations and predict a raw **PPG (photoplethysmography) signal**.
4.  **Signal Processing**:
    *   **Bandpass Filter**: Removes noise outside the typical heart rate range (0.7 - 3.0 Hz).
    *   **FFT (Fast Fourier Transform)**: Converts the PPG signal from time to frequency domain to find the dominant "pulse" frequency.
5.  **BPM & Stability**: The dominant frequency is converted to **BPM**. The Stability Manager rejects outliers and smooths the reading over time for a steady output.

## Deploying to Vercel

1.  **Connect Repo to Vercel**: Connect your GitHub repository to Vercel.
2.  **Configuration**: The project includes a `vercel.json` file to handle the Flask deployment.
3.  **Environment Variables**: Ensure `PORT` is not restricted (Vercel handles this automatically).
4.  **Statelessness Note**: Vercel Serverless Functions are **stateless**. The current in-memory `user_states` dictionary will be cleared when the function instance scales down or restarts. For production, you should replace this with a persistent store (e.g., Redis) or move inference to the client-side using ONNX Runtime Web.
5.  **Limits**: Serverless functions have a size limit (usually 250MB). PyTorch and MediaPipe can be heavy; if you exceed this limit, consider using a more lightweight inference engine or a different cloud provider (e.g., AWS Lambda with layers, or a dedicated VPS).

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

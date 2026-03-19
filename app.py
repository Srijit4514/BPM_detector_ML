import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
import torch
from src.model import PhysNetED
from face.detect import FaceDetector
from src.processor import SignalProcessor
from src.stability import StabilityManager
import base64

app = Flask(__name__)

# Components
detector = FaceDetector()
# We'll create processors and stability managers per session/client
# Use a simple dictionary for this demo; in production use Redis or similar.
user_states = {}

# Load model
DEVICE = torch.device("cpu")
MODEL_PATH = "models/physnet_single_debug.pth"
model = PhysNetED().to(DEVICE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    client_id = request.json.get('client_id', 'default_user')

    if client_id not in user_states:
        user_states[client_id] = {
            'frames_buffer': [],
            'ppg_signal': [],
            'processor': SignalProcessor(fps=10, window_size=64), # Sync with frontend 10 FPS
            'stability': StabilityManager()
        }

    state = user_states[client_id]

    data = request.json['image']
    data = data.split(',')[1]
    img_bytes = base64.b64decode(data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result = detector.extract_face(frame)
    bpm = 0
    quality = "Poor"

    if result is not None:
        face, forehead = result
        # Use forehead ROI if available, otherwise fallback to face
        roi = forehead if (forehead is not None and forehead.size > 0) else face
        face_normalized = cv2.resize(roi, (128, 128)) / 255.0
        state['frames_buffer'].append(face_normalized)

        if len(state['frames_buffer']) > 64:
            state['frames_buffer'].pop(0)

        if len(state['frames_buffer']) == 64:
            input_tensor = torch.FloatTensor(np.array(state['frames_buffer'])).permute(3, 0, 1, 2).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred_ppg = model(input_tensor).squeeze().cpu().numpy()
                state['ppg_signal'].append(float(pred_ppg[-1])) # Convert to float for JSON
                if len(state['ppg_signal']) > 128:
                    state['ppg_signal'].pop(0)

            if len(state['ppg_signal']) >= 64:
                raw_bpm, confidence = state['processor'].find_bpm(np.array(state['ppg_signal']))
                bpm = state['stability'].update_bpm(raw_bpm)
                quality = state['stability'].get_signal_quality(confidence)

    return jsonify({
        'bpm': f"{bpm:.1f}",
        'quality': quality,
        'signal': [float(x) for x in state['ppg_signal'][-50:]] if state['ppg_signal'] else []
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

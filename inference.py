import torch
import numpy as np
import matplotlib.pyplot as plt
from src.model import PhysNetED
from src.data_single import load_video_frames
import os

# ----------------------------
# Configuration (FIXED) Only for Google Colab
# ----------------------------
# VIDEO_PATH = "/content/physnet-colab/testing.mp4"
# MODEL_PATH = "/content/physnet-colab/models/physnet_single_debug.pth"
# OUTPUT_PATH = "/content/physnet-colab/predicted_ppg.npy"
# PLOT_RESULTS = True

# ----------------------------
# Configuration
# ----------------------------
VIDEO_PATH = "testing.mp4"
MODEL_PATH = "models/physnet_single_debug.pth"  # Path to your trained model
OUTPUT_PATH = "predicted_ppg.npy"         # Where to save predicted PPG signal
PLOT_RESULTS = True                        # Set to True to visualize results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# Check if model exists
# ----------------------------
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Trained model not found at {MODEL_PATH}")
    print("Please train the model first using train_single.py")
    exit(1)

# ----------------------------
# Load video frames
# ----------------------------
print(f"Loading video: {VIDEO_PATH}")

# Create a dummy PPG array (we don't need ground truth for inference)
dummy_ppg = np.zeros(1000)  # placeholder
frames, _ = load_video_frames(VIDEO_PATH, dummy_ppg)

print(f"Loaded {frames.shape[0]} frames from video")

# Convert to tensor: (N, H, W, C) -> (1, C, T, H, W)
frames_t = torch.FloatTensor(frames).permute(3, 0, 1, 2).unsqueeze(0)
print(f"Frame tensor shape: {frames_t.shape}")

# ----------------------------
# Load trained model
# ----------------------------
print(f"Loading model from: {MODEL_PATH}")
model = PhysNetED().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ----------------------------
# Run inference
# ----------------------------
print("Running inference...")
with torch.no_grad():
    pred_ppg = model(frames_t.to(device)).squeeze(0).cpu().numpy()

print(f"Predicted PPG shape: {pred_ppg.shape}")

# ----------------------------
# Save results
# ----------------------------
np.save(OUTPUT_PATH, pred_ppg)
print(f"Predicted PPG saved to: {OUTPUT_PATH}")

# ----------------------------
# Visualize results (optional)
# ----------------------------
if PLOT_RESULTS:
    plt.figure(figsize=(15, 4))
    plt.plot(pred_ppg, linewidth=0.8, color='blue')
    plt.title('Predicted rPPG Signal from testing.mp4')
    plt.xlabel('Frame')
    plt.ylabel('rPPG Amplitude')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = "predicted_ppg.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")
    plt.show()

print("\nInference complete!")

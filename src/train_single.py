import os
import time
import torch
import torch.optim as optim
from model import PhysNetED
from data_single import load_video_frames
from loss import neg_pearson_loss
import numpy as np

# Simple checkpoint helper for Colab (writes to Drive)
checkpoint_dir = "/content/drive/MyDrive/physnet/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)


def save_ckpt(model, optimizer, epoch, step, best_loss=None):
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = f"{checkpoint_dir}/physnet_e{epoch}_s{step}.pt"
    torch.save({
        "epoch": epoch,
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_loss": best_loss,
    }, path)
    print(f"saved {path}")
    return path

# For Google Colab users

# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# video_path = os.path.join(BASE_DIR, "testing.mp4")
# ppg_labels = np.load(os.path.join(BASE_DIR, "ppg_label.npy"))
# MODELS_DIR = os.path.join(BASE_DIR, "models")

# os.makedirs(MODELS_DIR, exist_ok=True)

# print("Video path:", video_path)
# print("PPG label path:", os.path.join(BASE_DIR, "ppg_label.npy"))
# print("Models directory:", MODELS_DIR)

# ----------------------------
# Load a small portion of video frames & labels
# ----------------------------
video_path = "../testing.mp4"
ppg_labels = np.load("../ppg_label.npy")  

# Load frames normally
frames, labels = load_video_frames(video_path, ppg_labels)

# Take only first N frames for fast debugging
N = 64  # small number of frames
frames = frames[:N]
labels = labels[:N]

# Convert to tensors
frames_t = torch.FloatTensor(frames).permute(3,0,1,2).unsqueeze(0)  # (1, C, T, H, W)
labels_t = torch.FloatTensor(labels).unsqueeze(0)                    # (1, T)

# ----------------------------
# Force CPU for debugging (optional, can use GPU too)
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # change to "cuda" if GPU available
print("Using device:", device)

# ----------------------------
# Create Model + Optimizer
# ----------------------------
model = PhysNetED().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ----------------------------
# Training Loop (very few epochs)
# ----------------------------
epochs = 2   # only 2 epochs for debugging
for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()

    # Forward
    pred = model(frames_t.to(device)).squeeze(0)   # (T)

    # Loss
    loss = neg_pearson_loss(pred.unsqueeze(0), labels_t.to(device))
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}/{epochs}   Loss: {loss.item():.5f}")

# ----------------------------
# Save trained checkpoint
# ----------------------------
save_path = "../models/physnet_single_debug.pth"
torch.save(model.state_dict(), save_path)
print("Debug model saved to", save_path)

# For Google Colab users
# save_path = os.path.join(MODELS_DIR, "physnet_single_debug.pth")
# torch.save(model.state_dict(), save_path)
# print("Debug model saved to", save_path)


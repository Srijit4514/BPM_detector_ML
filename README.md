# BPM_detector_ML

This repository contains a **video-based heart rate (BPM) detection pipeline** using a PhysNet-based deep learning model. It supports **training**, **inference**, **PPG generation**, and **BPM calculation** from facial videos.

The README only explains **how to use the existing files** (no new architecture or theory).

---

## Project Structure

```
BPM_detector_ML/
│
├── face/
│   └── detect.py              # Face detection / ROI extraction
│
├── models/
│   ├── physnet_e100_s0.pt      # Pretrained PhysNet weights
│   └── physnet_single_debug.pth# Trained / debug checkpoint
│
├── src/
│   ├── data_single.py          # Video frame + label loader
│   ├── loss.py                 # Negative Pearson loss
│   ├── model.py                # PhysNet model definition
│   └── train_single.py         # Training script
│
├── generate_fake_ppg.py        # Generate PPG signal from video
├── inference.py                # Run model inference on video
├── ppg_to_bpm.py               # Convert PPG → BPM
├── simple_print.py             # Simple test / sanity script
│
├── ppg_label.npy               # Ground-truth PPG (example)
├── predicted_ppg.npy           # Model output PPG
├── predicted_ppg.png           # Visualization of PPG
│
└── requirements.txt            # Python dependencies
```

---

## Installation

### 1. Clone repository

```bash
git clone https://github.com/Srijit4514/BPM_detector_ML.git
cd BPM_detector_ML
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Recommended**: Python 3.8+ and PyTorch with CUDA if available.

---

## Face Detection (Optional Preprocessing)

```bash
python face/detect.py --video input.mp4
```

* Extracts face region for better rPPG signal
* Can be skipped if videos are already cropped

---

## Training the Model

```bash
python src/train_single.py \
  --video path/to/video.mp4 \
  --label path/to/ppg_label.npy \
  --epochs 100
```

**Used files:**

* `src/model.py`
* `src/data_single.py`
* `src/loss.py`
* `models/physnet_e100_s0.pt` (optional pretrained)

Trained model checkpoints are saved as `.pth` files.

---

## Running Inference (PPG Prediction)

```bash
python inference.py \
  --video path/to/video.mp4 \
  --model models/physnet_single_debug.pth
```

**Outputs:**

* `predicted_ppg.npy`
* `predicted_ppg.png`

---

## Generate PPG Only

```bash
python generate_fake_ppg.py --video path/to/video.mp4
```

Useful when:

* Testing model behavior
* No ground-truth PPG available

---

## Convert PPG to BPM

```bash
python ppg_to_bpm.py --ppg predicted_ppg.npy
```

**Output:**

* Prints estimated **BPM (heart rate)**

---

## Quick Sanity Test

```bash
python simple_print.py
```

Checks:

* Model loading
* File paths
* Environment setup

---

## Model Files

| File                       | Purpose                    |
| -------------------------- | -------------------------- |
| `physnet_e100_s0.pt`       | Pretrained PhysNet weights |
| `physnet_single_debug.pth` | Fine-tuned / debug model   |

---

## Notes

* Works with `.mp4` videos
* GPU strongly recommended for training

---

## License

For research and educational use only.

import numpy as np
import argparse
from src.processor import SignalProcessor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def evaluate_performance(pred_ppg_path, gt_ppg_path, fps=30):
    """Evaluate rPPG prediction accuracy against ground truth."""
    pred_ppg = np.load(pred_ppg_path)
    gt_ppg = np.load(gt_ppg_path)

    # Ensure same length
    min_len = min(len(pred_ppg), len(gt_ppg))
    pred_ppg = pred_ppg[:min_len]
    gt_ppg = gt_ppg[:min_len]

    # Extract BPM values using sliding windows for comparison
    window_size = 64
    processor = SignalProcessor(fps=fps, window_size=window_size)
    step_size = 16

    pred_bpms = []
    gt_bpms = []

    for i in range(0, min_len - window_size, step_size):
        pred_window = pred_ppg[i:i+window_size]
        gt_window = gt_ppg[i:i+window_size]

        pred_bpm, _ = processor.find_bpm(pred_window)
        gt_bpm, _ = processor.find_bpm(gt_window)

        if pred_bpm > 0 and gt_bpm > 0:
            pred_bpms.append(pred_bpm)
            gt_bpms.append(gt_bpm)

    if not pred_bpms:
        print("Error: No valid BPM values extracted for evaluation.")
        return

    mae = mean_absolute_error(gt_bpms, pred_bpms)
    rmse = root_mean_squared_error(gt_bpms, pred_bpms)

    print(f"Evaluation Metrics:")
    print(f"  MAE (Mean Absolute Error): {mae:.2f} BPM")
    print(f"  RMSE (Root Mean Squared Error): {rmse:.2f} BPM")
    print(f"  Total windows compared: {len(pred_bpms)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate rPPG BPM accuracy.")
    parser.add_argument("--pred", type=str, default="predicted_ppg.npy", help="Path to predicted PPG signal.")
    parser.add_argument("--gt", type=str, default="ppg_label.npy", help="Path to ground truth PPG signal.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second.")

    args = parser.parse_args()
    evaluate_performance(args.pred, args.gt, args.fps)

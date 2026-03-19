import numpy as np

class StabilityManager:
    def __init__(self, window_size=5, outlier_threshold=15):
        self.bpm_history = []
        self.window_size = window_size
        self.outlier_threshold = outlier_threshold

    def update_bpm(self, current_bpm):
        """Update the BPM history and apply outlier rejection."""
        if current_bpm <= 0:
            return 0

        if not self.bpm_history:
            self.bpm_history.append(current_bpm)
            return current_bpm

        # Calculate a moving average
        avg_bpm = np.mean(self.bpm_history)

        # Reject outlier readings that deviate significantly from the moving average
        if abs(current_bpm - avg_bpm) > self.outlier_threshold:
            return avg_bpm

        self.bpm_history.append(current_bpm)
        if len(self.bpm_history) > self.window_size:
            self.bpm_history.pop(0)

        return np.mean(self.bpm_history)

    def get_signal_quality(self, confidence_score):
        """Determine the quality of the signal based on the confidence score."""
        if confidence_score > 0.8:
            return "Excellent"
        elif confidence_score > 0.5:
            return "Good"
        elif confidence_score > 0.3:
            return "Fair"
        else:
            return "Poor"

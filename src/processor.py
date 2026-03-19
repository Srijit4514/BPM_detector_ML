import numpy as np
from scipy import signal

class SignalProcessor:
    def __init__(self, fps=30, window_size=256, min_bpm=45, max_bpm=180):
        self.fps = fps
        self.window_size = window_size
        self.min_hz = min_bpm / 60
        self.max_hz = max_bpm / 60
        self.history = []

    def bandpass_filter(self, data):
        """Apply a Butterworth bandpass filter to the PPG signal."""
        nyquist = 0.5 * self.fps
        low = self.min_hz / nyquist
        high = self.max_hz / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        return signal.filtfilt(b, a, data)

    def detrend(self, data):
        """Removes the linear trend from the PPG signal."""
        return signal.detrend(data)

    def normalize(self, data):
        """Normalize the signal to [0, 1]."""
        return (data - np.mean(data)) / (np.std(data) + 1e-8)

    def find_bpm(self, ppg_signal):
        """Calculate BPM from PPG signal using FFT."""
        if len(ppg_signal) < self.window_size:
            return 0, 0.0

        ppg_filtered = self.bandpass_filter(self.detrend(ppg_signal))

        # Apply FFT to find the dominant frequency
        freqs = np.fft.fftfreq(len(ppg_filtered), d=1/self.fps)
        fft_vals = np.abs(np.fft.fft(ppg_filtered))

        # Only consider frequencies within the expected heart rate range
        mask = (freqs >= self.min_hz) & (freqs <= self.max_hz)
        freqs = freqs[mask]
        fft_vals = fft_vals[mask]

        if len(fft_vals) == 0:
            return 0, 0.0

        peak_idx = np.argmax(fft_vals)
        peak_freq = freqs[peak_idx]
        bpm = peak_freq * 60

        # Confidence score: peak height vs. total energy
        confidence = fft_vals[peak_idx] / (np.sum(fft_vals) + 1e-8)

        return bpm, confidence

    def moving_average(self, data, window=5):
        """Smooths signal with a simple moving average."""
        return np.convolve(data, np.ones(window)/window, mode='same')

    def temporal_smoothing(self, current_bpm, last_bpm, alpha=0.3):
        """Alpha smoothing for the output BPM."""
        if last_bpm == 0:
            return current_bpm
        return alpha * current_bpm + (1 - alpha) * last_bpm

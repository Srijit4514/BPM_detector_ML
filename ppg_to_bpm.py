import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# For Google Colab:
# ppg = np.load("/content/physnet-colab/predicted_ppg.npy")

# Load the predicted PPG
ppg = np.load("ppg_label.npy")

# Assume 30 FPS video (adjust based on your actual video FPS)
fps = 30  

# Apply bandpass filter (typical heart rate 0.7-4 Hz = 42-240 BPM)
sos = signal.butter(4, [0.7, 4], 'bandpass', fs=fps, output='sos')
ppg_filtered = signal.sosfiltfilt(sos, ppg)

# Compute FFT to find dominant frequency
freqs = np.fft.fftfreq(len(ppg_filtered), d=1/fps)
fft_vals = np.abs(np.fft.fft(ppg_filtered))

# Only positive frequencies
positive_freqs = freqs[:len(freqs)//2]
positive_fft = fft_vals[:len(fft_vals)//2]

# Find peak frequency
peak_idx = np.argmax(positive_fft)
peak_freq = positive_freqs[peak_idx]
bpm = peak_freq * 60  # Convert Hz to BPM

print(f"Detected BPM: {bpm:.1f}")

# Visualize
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(ppg_filtered)
plt.title("Filtered PPG Signal")
plt.xlabel("Frame")
plt.ylabel("Amplitude")
plt.subplot(1, 2, 2)
plt.plot(positive_freqs, positive_fft)
plt.title("FFT Spectrum")
plt.xlabel("Frequency (Hz)")
plt.axvline(peak_freq, color='r', label=f'Peak: {bpm:.1f} BPM')
plt.legend()
plt.tight_layout()
plt.show()
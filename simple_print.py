import numpy as np

# Load the .npy file
ppg_labels = np.load("ppg_label.npy")
# predicted_ppg = np.load("predicted_ppg.npy")

# Print info
# print(ppg_labels)          # prints all values (if small)
# print(ppg_labels.shape)    # prints the shape of the array
# print(ppg_labels.dtype)    # prints the data type

# print(predicted_ppg)          # prints all values (if small)
# print(predicted_ppg.shape)    # prints the shape of the array
# print(predicted_ppg.dtype)    # prints the data type

import matplotlib.pyplot as plt

plt.plot(ppg_labels)
# plt.plot(predicted_ppg)
plt.title("PPG Signal from .npy")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

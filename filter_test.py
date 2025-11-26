import frequency_decomposition
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np

notch_filter = frequency_decomposition.NotchFilter(50, 1000)
freq, h = signal.freqz(notch_filter.b, notch_filter.a, fs=1000)

# Plot
fig, ax = plt.subplots(2, 1, figsize=(8, 6))
ax[0].plot(freq, 20*np.log10(abs(h)), color='blue')
ax[0].set_title("Frequency Response")
ax[0].set_ylabel("Amplitude [dB]", color='blue')
ax[0].set_xlim([0, 100])
ax[0].set_ylim([-25, 10])
ax[0].grid(True)
ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='green')
ax[1].set_ylabel("Phase [deg]", color='green')
ax[1].set_xlabel("Frequency [Hz]")
ax[1].set_xlim([0, 100])
ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
ax[1].set_ylim([-90, 90])
ax[1].grid(True)
plt.show()

from frequency_decomposition import FrequencyDecomposition
import matplotlib.pyplot as plt
import cupyx.scipy.signal as signal
import numpy as np

freq_decomposer = FrequencyDecomposition(32, 1000, 4, 8, 35)
freqzs = [signal.sosfreqz(freq_decomposer.sos[i], fs=1000) for i in range(freq_decomposer.num_frequency_bands)]

# Frequency response
fig, axes = plt.subplots(2, 1)

# Plot Magniture Response
for i in range(len(freqzs)):
    w, h, = freqzs[i]
    w = w.get()
    h = h.get()
    db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
    axes[0].plot(w, db)
axes[0].grid(True)
axes[0].set_ylim(-4, 2)
axes[0].set_yticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
axes[0].set_xlim(0, 40)
axes[0].set_xticks([2*i for i in range(1, 21)])
axes[0].set_ylabel('Gain [dB]')
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_title('Magnitude Response')

# Plot Phase Response
for i in range(len(freqzs)):
    w, h = freqzs[i]
    w = w.get()
    h = h.get()
    phase = np.angle(h)
    axes[1].plot(w, phase)
axes[1].grid(True)
axes[1].set_xlim(0, 40)
axes[1].set_xticks([6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36])
axes[1].set_yticks([-np.pi, -0.5 * np.pi, 0, 0.5 * np.pi, np.pi])
axes[1].set_yticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
axes[1].set_ylabel('Phase (rad)')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_title('Phase Response')

# Add overall title and adjust layout
fig.suptitle('Frequency Response', fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])


plt.show()


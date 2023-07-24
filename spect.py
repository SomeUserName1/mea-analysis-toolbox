import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Generate data
fs = 10e2
N = 5e6
noise_power = 0.01 * fs / 2
time = np.arange(N) / float(fs)
noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
noise *= np.exp(-time/5)
xs = np.empty((5, int(N)))
amp = 4 * np.sqrt(2)
mods = [0, 200, 400, 600, 800]
for i, mod_i in enumerate(mods):
    mod = mod_i*np.cos(2*np.pi*0.2*time)
    carrier = amp * np.sin(2*np.pi*time + mod)
    xs[i] = carrier + noise

# matplotlib

NFFT=int(fs)
print(NFFT)
mpl_specgram_window = plt.mlab.window_hanning(np.ones(NFFT))
f, t, Sxxs = signal.spectrogram(xs, fs, detrend=False,
                                   nfft=NFFT, 
                                   window=mpl_specgram_window, 
                                  )
for i, _ in enumerate(mods):
    plt.figure()
    Pxx, freqs, bins, im = plt.specgram(xs[i], NFFT=NFFT, Fs=fs)
    plt.axis((None, None, 0, 200))
    plt.show(block=False)

    plt.figure()
    plt.pcolormesh(t, f, Sxxs[i])
    plt.axis((None, None, 0, 200))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


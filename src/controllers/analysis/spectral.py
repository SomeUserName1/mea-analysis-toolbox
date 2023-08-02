from fooof import FOOOF
import numba as nb
import numpy as np
import scipy.signal as sg

from model.data import Data

# No njit as numpy.fft is not supported & numpy already calls C routines
def compute_psds(data: Data) -> tuple[np.ndarray, np.ndarray]:
    ys = data.data
    fft = np.fft.rfft(ys)
    power = 2 / np.square(ys.shape[1]) * (fft.real**2 + fft.imag**2)
    freq = np.fft.rfftfreq(ys.shape[1], 1/data.sampling_rate)
    phase = np.angle(fft)
    phase[np.abs(fft) < 1] = 0

    data.psds = freq, power, phase


# No njit as fooof is unknown to numba
def compute_periodic_aperiodic_decomp(data: Data,
                                      freq_range: tuple[int, int]=(1, 150)
                                      ) -> FOOOFGroup:
    if data.psds is None:
        compute_psds(data.data)

    fg = FOOOFGroup()
    fg.fit(data.psds, freq_range, n_jobs=-1)
    data.fooof_group = fg


# will probably not work with numba.
# alternatively parallelize or extract fooof components into matrix and do
# subtraction as vectrorized (in a separate fn with numba)
# alternatively parallelize and collect.
# prolly IO bound
@nb.jit(parallel=True)
def detrend_fooof(data: Data):
    if data.fooof_group is None:
        fg = compute_periodic_aperiodic_decomp(data)

    n_els = data.data.shape[0]
    norm_psd = np.empty((n_els,
                         fg.get_fooof(0).power_spectrum))
    for idx in range(n_els):
        fooof = fg.get_fooof(idx)

        if fooof.has_model:
            norm_psd[idx] = fooof.power_spectrum - fooof._ap_fit
        else:
            norm_psd[idx] = None

    data.detrended_psds = norm_psd


# No njit as scipy.signal is not supported & scipy already calls C routines
def compute_spectrograms(data: Data):
    data.spectrograms =  sg.spectrogram(data.data, data.sampling_rate,
                                          nfft=1024)




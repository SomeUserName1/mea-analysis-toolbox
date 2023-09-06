"""
TODO
"""
from fooof import FOOOFGroup
# import numba as nb
import numpy as np
import scipy.signal as sg

from model.data import Recording, SharedArray


# No njit as numpy.fft is not supported & numpy already calls C routines
def compute_psds(rec: Recording):
    """
    Compute the power spectral density of the data in the Recording object.

    :param rec: The recording object.
    :type rec: Recording
    """
    ys = rec.get_data()
    fft = np.fft.rfft(ys)
    power = 2 / np.square(ys.shape[1]) * (fft.real**2 + fft.imag**2)
    freq = np.fft.rfftfreq(ys.shape[1], 1 / rec.sampling_rate)
    phase = np.angle(fft)
    phase[np.abs(fft) < 1] = 0

    rec.psds = SharedArray(freq), SharedArray(power), SharedArray(phase)


# No njit as fooof is unknown to numba
def compute_periodic_aperiodic_decomp(rec: Recording,
                                      freq_range: tuple[int, int] = (1, 150)
                                      ):
    """
    Compute the periodic and aperiodic decomposition of the power spectral
    density of the data in the Recording object.

    :param rec: The recording object.
    :type rec: Recording

    :param freq_range: The frequency range to fit the periodic and aperiodic
                       parts of the power spectral density.
    :type freq_range: tuple[int, int]
    """
    if rec.psds is None:
        compute_psds(rec)

    fg = FOOOFGroup()
    fg.fit(rec.psds, freq_range, n_jobs=-1)
    rec.fooof_group = fg


# will probably not work with numba.
# alternatively parallelize or extract fooof components into matrix and do
# subtraction as vectrorized (in a separate fn with numba)
# alternatively parallelize and collect.
# prolly IO bound
# @nb.njit(parallel=True)
def detrend_fooof(rec: Recording):
    """
    Detrend the power spectral density of the data in the Recording object
    using the periodic and aperiodic decomposition.

    :param rec: The recording object.
    :type rec: Recording
    """
    if rec.fooof_group is None:
        compute_periodic_aperiodic_decomp(rec)

    fg = rec.fooof_group
    n_els = len(rec.selected_electrodes)
    norm_psd = np.empty((n_els, fg.get_fooof(0).power_spectrum))
    for idx in range(n_els):
        fooof = fg.get_fooof(idx)

        if fooof.has_model:
            norm_psd[idx] = fooof.power_spectrum - fooof._ap_fit
        else:
            norm_psd[idx] = None

    rec.detrended_psds = norm_psd


# No njit as scipy.signal is not supported & scipy already calls C routines
def compute_spectrograms(rec: Recording):
    """
    Compute the spectrograms of the data in the Recording object.

    :param rec: The recording object.
    :type rec: Recording
    """
    rec.spectrograms = sg.spectrogram(rec.get_data(), rec.sampling_rate,
                                      nfft=1024)

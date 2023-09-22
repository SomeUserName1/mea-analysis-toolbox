"""
TODO
"""
from fooof import FOOOFGroup
import numpy as np
import scipy.signal as sg
import pdb

from model.data import Recording, SharedArray
from constants import default_bins


# No njit as numpy.fft is not supported & numpy already calls C routines
def compute_psds_non_smooth(rec: Recording):
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

    rec.psds = SharedArray(freq), SharedArray(power)


# No njit as numpy.fft is not supported & numpy already calls C routines
def compute_psds(rec: Recording):
    """
    Compute the power spectral density of the data in the Recording object.

    :param rec: The recording object.
    :type rec: Recording
    """
    ys = rec.get_data()
    freq, power = sg.welch(ys, fs=rec.sampling_rate, nperseg=256, nfft=512)
    rec.psds = SharedArray(freq), SharedArray(power)


# No njit as scipy.signal is not supported & scipy already calls C routines
def compute_spectrograms(rec: Recording):
    """
    Compute the spectrograms of the data in the Recording object.

    :param rec: The recording object.
    :type rec: Recording
    """
    win = np.kaiser(128, 0)
    f, t, sxx = sg.spectrogram(rec.get_data(), rec.sampling_rate,
                               window=win, nperseg=len(win),
                               noverlap=len(win) / 4, nfft=2 * len(win))
    rec.spectrograms = SharedArray(f), SharedArray(t), SharedArray(sxx)


    freq_bin_names = [f"{bins[0]}-{bins[1]}" for bins in default_bins]
    binned_power = []
    for idx in range(len(rec.selected_electrodes)):
        binned_power.append(bin_powers(rec, idx, (rec.start_idx, rec.stop_idx)))

    rec.channels_df[freq_bin_names] = binned_power



def bin_powers(rec, el_idx, idx_range, bin_ranges=default_bins):
    """
    Sum the power per frequency bin.

    :param rec: The recording object.
    :type rec: Recording

    :param idx_range: The range of time indexes to sum the power for.
    :type idx_range: tuple[int, int]

    :param bin_range: List of frequency ranges to sum the power in.
    :type bin_range: list[tuple[int, int]]

    :return: A list of the sum of the powers per frequency bin.
    :rtype: list[float]
    """
    freqs = rec.spectrograms[0].read()
    t_start = idx_range[0] / rec.sampling_rate
    t_stop = idx_range[1] / rec.sampling_rate
    times = rec.spectrograms[1].read()
    times = np.argwhere((times >= t_start) & (times < t_stop))
    power = rec.spectrograms[2].read()
    bin_powers = np.empty(len(bin_ranges))

    for idx, bin_range in enumerate(bin_ranges):
        bin_freqs = np.argwhere(((freqs >= bin_range[0]) & (freqs < bin_range[1])))
        bin_pow = power[el_idx, bin_freqs, :]
        bin_pow = bin_pow[:, :, times]
        bin_powers[idx] = np.sum(bin_pow) / (times.shape[0] * bin_freqs.shape[0])

    return bin_powers


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


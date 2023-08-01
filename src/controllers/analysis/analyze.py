import time
from multiprocessing import Process

import antropy as ant
import elephant as ep
from fooof import FOOOF
import minfo
import numba as nb
import numpy as np
from PyIF.te_compute import te_compute
import scipy.signal as sg

from model.Event import Event


@nb.njit(parallel=True)
def compute_snrs(result: Result):
    signals = result.data
    result.snrs = np.square(np.mean(signals, axis=-1)) / np.var(signals, axis=-1)


@nb.njit(parallel=True)
def compute_rms(result: Result) -> np.ndarray:
    result.rms = np.sqrt(np.mean(np.square(result.data), axis=-1))


@nb.njit(parallel=True)
def compute_derivative(result: Result):
    result.derivatives = np.diff(result.data) * result.sampling_rate # x / 1 /sampling period == x * sampling_period


def compute_mv_avg(result: Result, w: int=None):
    result.mv_means = moving_avg(result.data, w, fs=result.sampling_rate)


@nb.njit(parallel=True)
def moving_avg(sig: np.ndarray, w: int=None, fs: int) -> np.ndarray:
    if w is None:
        w = int(np.round(0.01 * fs)) # 10 ms
    if w % 2 == 0
        w = w + 1

    pad = (w - 1) / 2
    abs_pad = np.pad(np.absolute(sig), (pad, pad), "edge")
    ret = np.cumsum(abs_pad, dtype=float, axis=-1)
    ret[:, w:] = ret[:, w:] - ret[:, :-w]

    return ret[w - 1:] / w


@nb.njit(parallel=True)
def compute_mv_var(result: Result, w: int=None):
    sigs = result.data
    sq_dev = np.square((sigs.T - np.mean(sigs, axis=-1)).T)
    result.mv_vars = moving_avg(sq_dev, w=w, fs=result.sampling_rate)


@nb.njit(parallel=True)
def compute_mv_mads(result: Result, w: int=None):
    sigs = result.data
    abs_devs = np.absolute((sigs.T - np.mean(sigs, axis=-1)).T)
    result.mv_mads = moving_avg(abs_dev, w=w, fs=result.sampling_rate)


# No njit as scipy.signal is not supported & scipy already calls C routines
def compute_envelopes(result: Result):
    result.envelopes = np.absolute(sg.hilbert(result.data))


# No njit as numpy.fft is not supported & numpy already calls C routines
def compute_psds(result: Result) -> tuple[np.ndarray, np.ndarray]:
    ys = result.data
    fft = np.fft.rfft(ys)
    power = 2 / np.square(ys.shape[1]) * (fft.real**2 + fft.imag**2)
    freq = np.fft.rfftfreq(ys.shape[1], 1/result.sampling_rate)
    phase = np.angle(fft)

    result.psds = freq, power, phase


# No njit as fooof is unknown to numba
def compute_periodic_aperiodic_decomp(result: Result,
                                      freq_range: tuple[int, int]=(1, 150)
                                      ) -> FOOOFGroup:
    if result.psds is None:
        compute_psds(result.data)

    fg = FOOOFGroup()
    fg.fit(result.psds, freq_range, n_jobs=-1)
    result.fooof_group = fg


# will probably not work with numba.
# alternatively parallelize or extract fooof components into matrix and do
# subtraction as vectrorized (in a separate fn with numba)
# alternatively parallelize and collect.
# prolly IO bound
@nb.jit(parallel=True)
def detrend_fooof(result: Result):
    if result.fooof_group is None:
        fg = compute_periodic_aperiodic_decomp(result)

    n_els = result.data.shape[0]
    norm_psd = np.empty((n_els,
                         fg.get_fooof(0).power_spectrum))
    for idx in range(n_els):
        fooof = fg.get_fooof(idx)

        if fooof.has_model:
            norm_psd[idx] = fooof.power_spectrum - fooof._ap_fit
        else:
            norm_psd[idx] = None

    result.detrended_psds = norm_psd


# No njit as scipy.signal is not supported & scipy already calls C routines
def compute_spectrograms(result: Result):
    result.spectrograms =  sg.spectrogram(result.data, result.sampling_rate,
                                          nfft=1024)


# antropy uses numba
# Parallelize for loop w. conc.fut.ProcessPoolExec maybe
@nb.jit(parallel=True)
def compute_entropies(result: Result):
    n_els = result.data.shape[0]
    entropies = np.zeros(n_els)
    for i in range(n_els):
        entropies[i] = ant.app_entropy(result.data[i])

# GENERAL
# baseline how?
# just not use it? in bursting signals the mean will be very high, so will then be the threshold if 3*std is applied
# just the first n secs? will not work with older recordings maybe
# use baseline recording? loads the whole file into memory and takes a lot of time
#
# RN used is the signal + 3 *the MAD of the signal. Alternative: use roling mad instead of signal
#
# ---find_peaks---
# seems to detect some peaks multiple times (check with higher resolution viz)
# prominences are not always correct?
#
# Maybe parallelize using ProcessPoolExec.
@nb.jit(parallel=True)
def detect_peaks(result: Result):
    signals = result.data
    mads = np.mean(np.absolute((signals.T - np.means(signals, axis=-1)).T), axis=-1)
    signals = np.absolute(signals)
    result.peaks = []

    for i in range(result.data.shape[0]):
        result.peaks.append(sg.find_peaks(signals[i], threshold=3*mads[i]))


@nb.njit(parallel=True)
def compute_inter_peak_intervals(result:Result):
    if result.peaks is None:
        compute_peaks(results)

    result.ipis = []
    for peaks, _ in result.peaks:
        ipi = np.diff(peaks)
        result.ipis.append(ipi)


# parallelize, maybe pull out z-scoring for numba
def compute_xcorrs(result: Result):
    sig = result.data
    means = np.mean(sig, axis=-1)
    stds = np.std(sig, axis=-1)
    sig = ((sig.T - means) / stds).T

    lags = sg.correlation_lags(sig.shape[1], sig.shape[1], "same")
    result.xcorrs = lags, np.zeros(sig.shape[0], sig.shape[0], sig.shape[1])

    for i, sig1 in enumerate(sig):
        for j, sig2 in enumerate(sig):
            elif i < j:
                continue
            else:
                result.xcorrs[1][i, j] = (sg.correlate(sig1, sig2, 'same')
                                        * (1 / (sig.shape[1] - np.abs(lags))))

    for i in range(sig.shape[0]):
        for j in range(sig.shape[0]):
            if i >= j:
                continue
            else:
                result.xcorrs[1][i, j] = result.xcorrs[1][j, i]

##### for the even more expensive stuff involving time lags, consider epoching

# parallelizing with con.fut.ProcessPoolExecutor
def compute_mutual_info(result: Result):
    for i, sig1 in enumerate(result.data):
        for j, sig2 in enumerate(result.data):
            if i < j:
                continue
            else:
                result.mutual_informations[i, j] = minfo.mi_float.mutual_info(
                        sig1, sig2, algorithm='adaptive')

    for i in range(n_channels):
        for j in range(n_channels):
            if i >= j:
                continue
            else:
                result.mutual_infos[i, j] = result.mutual_infos[j, i]


# PyIF.compute_te already uses numba/gpu
# Parallelizing the loop may be an option depending on the load
# concurrent.futures.ProcessPoolExecutor
@nb.jit(parallel=True)
def compute_transfer_entropy(result: Result, lag_ms: int = 1000):
    n_els = result.data.shape[0]
    result.transfer_entropies = np.zeros((n_els, n_els))

    lags = int(lag_ms * 0.001 * sampling_rate)
    if lags < 1:
        lags = 1

    for i, sig1 in enumerate(result.data):
        for j, sig2 in enumerate(result.data):
            if i == j:
               continue
            else:
                result.transfer_entropies[i, j] = compute_te(sig1, 
                                                             sig2,
                                                             embedding=lags)


def compute_coherence(result: Result):
    freqs = None
    coherences = []
    for i, sig1 in enumerate(result.data):
        coherences.append([])
        for j, sig2 in enumerate(result.data):
            if i < j:
                continue
            else:
                if freqs is None:
                    freqs, coh, lag = ep.spectral.multitaper_coherence(
                        sig1,
                        sig2,
                        fs=sampling_rate)
                else:
                    _, coh, lag = ep.spectral.multitaper_coherence(
                        sig1,
                        sig2,
                        fs=sampling_rate)
                coherences[i].append((coh, lags))

    result.coherences = freqs, coherences


def compute_granger_causality(result: Result, lag_ms=1000):
    lags = int(lag_ms * 0.001 * sampling_rate)
    cgs = []
    for i, sig1 in enumerate(result.data):
        cgs.append([])
        for j, sig2 in enumerate(result.data):
            if i < j:
                continue
            else:
                caus = ep.causality.granger.pairwise_granger(
                    np.vstack((sig1, sig2)),
                    max_order=lags)
            cgs[i].append(caus)

    result.granger_causalities = cgs


def compute_spectral_granger(result: Result):
    freqs = None
    coherences = []
    for i, sig1 in enumerate(result.data):
        spectrag_cgs.append([])
        for j, sig2 in enumerate(result.data):
            if i < j:
                continue
            else:
                if freqs is None:
                    freqs, scg = ep.causality.granger.pairwise_spectral_granger(
                        sig1,
                        sig2,
                        fs=sampling_rate)
                else:
                    _, scg = ep.causality.granger.pairwise_spectral_granger(
                        sig1,
                        sig2,
                        fs=sampling_rate)
                spectral_cgs[i].append(scg)

    result.coherences = freqs, spectral_cgs


def compute_current_source_density(result: Result):
    # TODO impl
    # use kCSD as only partial data is there and we often have broken
    #  electrodes which have to be cleaned/excluded beforehand. 
    coords = [[int(s)-1*200*pq.um * for s in name.split() if s.isdigit()] \
                      for name in result.names()]

    signal = neo.AnalogSignal(result.data, units='uV',
                              sampling_rate=result.sampling_rate*pq.Hz)

    result.csd = ep.current_source_density.estimate_csd(signal,
                                                        coords,
                                                        'KCSD2D')


def compute_phase_slope_index(result: Result):
    # TODO impl
    pass


def compute_phase_amplitude_coupling(result: Result):
    # TODO impl
    pass


@nb.njit
def bin_amplitude(result: Result, new_sr: int=500, absolute: bool=False) -> np.ndarray:
    signals = result.data
    n_bins = new_sr / result.sampling_rate * signals.shape[1]
    bins = np.zeros((signals.shape[0], n_bins))

    signal_idx = 0
    bin_idx = 0
    while signal_idx < result.data.shape[1]:
        t_int = 0
        n_frames_in_bin = 0

        while t_int < 1 / new_sr and signal_idx < result.data.shape[1]:
            bins[:, bin_idx] = bins[:, bin_idx] + result.data[:, signal_idx]

            t_int = t_int + 1 / result.sampling_rate
            signal_idx = signal_idx + 1
            n_frames_in_bin = n_frames_in_bin + 1

        bins[:, bin_idx] = bins[:, bin_idx] / n_frames_in_bin
        bin_idx = bin_idx + 1

    bins = np.array(bins)


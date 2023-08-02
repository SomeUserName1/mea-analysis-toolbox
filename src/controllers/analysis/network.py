import elephant as ep
import minfo
import numba as nb
import numpy as np
from PyIF.te_compute import te_compute
import scipy.signal as sg

from model.event import Event
from model.data import Data


# parallelize, maybe pull out z-scoring for numba
def compute_xcorrs(data: Data):
    sig = data.data
    means = np.mean(sig, axis=-1)
    stds = np.std(sig, axis=-1)
    sig = ((sig.T - means) / stds).T

    lags = sg.correlation_lags(sig.shape[1], sig.shape[1], "same")
    data.xcorrs = lags, np.zeros(sig.shape[0], sig.shape[0], sig.shape[1])

    for i, sig1 in enumerate(sig):
        for j, sig2 in enumerate(sig):
            elif i < j:
                continue
            else:
                data.xcorrs[1][i, j] = (sg.correlate(sig1, sig2, 'same')
                                        * (1 / (sig.shape[1] - np.abs(lags))))

    for i in range(sig.shape[0]):
        for j in range(sig.shape[0]):
            if i >= j:
                continue
            else:
                data.xcorrs[1][i, j] = data.xcorrs[1][j, i]

##### for the even more expensive stuff involving time lags, consider epoching

# parallelizing with con.fut.ProcessPoolExecutor
def compute_mutual_info(data: Data):
    for i, sig1 in enumerate(data.data):
        for j, sig2 in enumerate(data.data):
            if i < j:
                continue
            else:
                data.mutual_informations[i, j] = minfo.mi_float.mutual_info(
                        sig1, sig2, algorithm='adaptive')

    for i in range(n_channels):
        for j in range(n_channels):
            if i >= j:
                continue
            else:
                data.mutual_infos[i, j] = data.mutual_infos[j, i]


# PyIF.compute_te already uses numba/gpu
# Parallelizing the loop may be an option depending on the load
# concurrent.futures.ProcessPoolExecutor
@nb.jit(parallel=True)
def compute_transfer_entropy(data: Data, lag_ms: int = 1000):
    n_els = data.data.shape[0]
    data.transfer_entropies = np.zeros((n_els, n_els))

    lags = int(lag_ms * 0.001 * sampling_rate)
    if lags < 1:
        lags = 1

    for i, sig1 in enumerate(data.data):
        for j, sig2 in enumerate(data.data):
            if i == j:
               continue
            else:
                data.transfer_entropies[i, j] = compute_te(sig1, 
                                                             sig2,
                                                             embedding=lags)

# parallelize
def compute_coherence(data: Data):
    freqs = None
    coherences = []
    for i, sig1 in enumerate(data.data):
        coherences.append([])
        for j, sig2 in enumerate(data.data):
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

    data.coherences = freqs, coherences


# parallelize
def compute_granger_causality(data: Data, lag_ms=1000):
    lags = int(lag_ms * 0.001 * sampling_rate)
    cgs = []
    for i, sig1 in enumerate(data.data):
        cgs.append([])
        for j, sig2 in enumerate(data.data):
            if i < j:
                continue
            else:
                caus = ep.causality.granger.pairwise_granger(
                    np.vstack((sig1, sig2)),
                    max_order=lags)
            cgs[i].append(caus)

    data.granger_causalities = cgs


# parallelize
def compute_spectral_granger(data: Data):
    freqs = None
    coherences = []
    for i, sig1 in enumerate(data.data):
        spectrag_cgs.append([])
        for j, sig2 in enumerate(data.data):
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

    data.coherences = freqs, spectral_cgs


def compute_current_source_density(data: Data):
    coords = [[int(s)-1*200*pq.um for s in name.split() if s.isdigit()] \
                      for name in data.names()]

    signal = neo.AnalogSignal(data.data, units='uV',
                              sampling_rate=data.sampling_rate*pq.Hz)

    data.csd = ep.current_source_density.estimate_csd(signal,
                                                        coords,
                                                        'KCSD2D')


def compute_phase_slope_index(data: Data):
    pass


def compute_phase_amplitude_coupling(data: Data):
    pass


@nb.njit
def bin_amplitude(data: Data, new_sr: int=500, absolute: bool=False) -> np.ndarray:
    signals = data.data
    n_bins = new_sr / data.sampling_rate * signals.shape[1]
    bins = np.zeros((signals.shape[0], n_bins))

    signal_idx = 0
    bin_idx = 0
    while signal_idx < data.data.shape[1]:
        t_int = 0
        n_frames_in_bin = 0

        while t_int < 1 / new_sr and signal_idx < data.data.shape[1]:
            bins[:, bin_idx] = bins[:, bin_idx] + data.data[:, signal_idx]

            t_int = t_int + 1 / data.sampling_rate
            signal_idx = signal_idx + 1
            n_frames_in_bin = n_frames_in_bin + 1

        bins[:, bin_idx] = bins[:, bin_idx] / n_frames_in_bin
        bin_idx = bin_idx + 1

    bins = np.array(bins)


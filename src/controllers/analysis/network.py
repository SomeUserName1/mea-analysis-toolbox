"""
TODO
"""
import elephant as ep
from elephant.causality.granger import (pairwise_granger,
                                        pairwise_spectral_granger)
from mutual_info.mutual_info import mutual_information
# import numba as nb
import neo
import numpy as np
from PyIF.te_compute import te_compute
import scipy.signal as sg
import quantities as pq

# from src.model.event import Event
from model.data import Data


# parallelize, maybe pull out z-scoring for numba
def compute_xcorrs(data: Data):
    sig = data.data
    means = np.mean(sig, axis=-1)
    stds = np.std(sig, axis=-1)
    sig = ((sig.T - means) / stds).T

    lags = sg.correlation_lags(sig.shape[1], sig.shape[1], "same")
    data.xcorrs = lags, np.zeros((sig.shape[0], sig.shape[0], sig.shape[1]))

    for i, sig1 in enumerate(sig):
        for j, sig2 in enumerate(sig):
            if i < j:
                continue

            data.xcorrs[1][i, j] = (sg.correlate(sig1, sig2, 'same')
                                    * (1 / (sig.shape[1] - np.abs(lags))))

    for i in range(sig.shape[0]):
        for j in range(sig.shape[0]):
            if i >= j:
                continue

            data.xcorrs[1][i, j] = data.xcorrs[1][j, i]


# for the even more expensive stuff involving time lags, consider epoching
# parallelizing with con.fut.ProcessPoolExecutor
def compute_mutual_info(data: Data):
    for i, sig1 in enumerate(data.data):
        for j, sig2 in enumerate(data.data):
            if i < j:
                continue

            data.mutual_informations[i, j] = mutual_information((sig1, sig2))

    n_channels = len(data.selected_electrodes)
    for i in range(n_channels):
        for j in range(n_channels):
            if i >= j:
                continue

            data.mutual_infos[i, j] = data.mutual_infos[j, i]


# PyIF.compute_te already uses numba/gpu
# Parallelizing the loop may be an option depending on the load
# concurrent.futures.ProcessPoolExecutor
def compute_transfer_entropy(data: Data, lag_ms: int = 1000):
    n_els = data.data.shape[0]
    data.transfer_entropies = np.zeros((n_els, n_els))
    lags = int(lag_ms * 0.001 * data.sampling_rate)
    lags = max(lags, 1)

    for i, sig1 in enumerate(data.data):
        for j, sig2 in enumerate(data.data):
            if i == j:
                continue

            data.transfer_entropies[i, j] = te_compute(sig1,
                                                       sig2,
                                                       embedding=lags)


# parallelize
def compute_coherence(data: Data):
    coherences = []
    for i, sig1 in enumerate(data.data):
        coherences.append([])
        for j, sig2 in enumerate(data.data):
            if i < j:
                continue

            coh = ep.spectral.multitaper_coherence(sig1.T, sig2.T,
                                                   fs=data.sampling_rate)
            coherences[i].append(coh)

    data.coherences = np.array(coherences)


# parallelize
def compute_granger_causality(data: Data, lag_ms=1000):
    lags = int(lag_ms * 0.001 * data.sampling_rate)
    cgs = []
    for i, sig1 in enumerate(data.data):
        cgs.append([])
        for j, sig2 in enumerate(data.data):
            if i < j:
                continue
            caus = pairwise_granger(np.hstack((sig1.T, sig2.T)),
                                    max_order=lags)
            cgs[i].append(caus)

    data.granger_causalities = cgs


# parallelize
def compute_spectral_granger(data: Data):
    spectral_cgs = []
    for i, sig1 in enumerate(data.data):
        spectral_cgs.append([])
        for j, sig2 in enumerate(data.data):
            if i < j:
                continue

            scgs = pairwise_spectral_granger(sig1.T, sig2.T,
                                             fs=data.sampling_rate)
            spectral_cgs[i].append(scgs)

    data.spectal_granger = spectral_cgs


def compute_current_source_density(data: Data):
    coords = [[(int(s)-1) * 200 * pq.um for s in name.split() if s.isdigit()]
              for name in data.names()]

    signal = neo.AnalogSignal(data.data, units='V',
                              sampling_rate=data.sampling_rate*pq.Hz)

    data.csd = ep.current_source_density.estimate_csd(signal,
                                                      coords,
                                                      'KCSD2D')


def compute_phase_slope_index(data: Data):
    pass


def compute_phase_amplitude_coupling(data: Data):
    pass

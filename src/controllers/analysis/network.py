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
from model.data import Recording


# parallelize, maybe pull out z-scoring for numba
def compute_xcorrs(rec: Recording):
    """
    Compute the cross-correlation of the data in the Recording object.
    First the signal is z-scored. The cross-correlation is then computed using
    the scipy.signal.correlate function.

    :param rec: The recording object.
    :type rec: Recording
    """
    sig = rec.get_data()
    means = np.mean(sig, axis=-1)
    stds = np.std(sig, axis=-1)
    sig = ((sig.T - means) / stds).T

    lags = sg.correlation_lags(sig.shape[1], sig.shape[1], "same")
    rec.xcorrs = lags, np.zeros((sig.shape[0], sig.shape[0], sig.shape[1]))

    for i, sig1 in enumerate(sig):
        for j, sig2 in enumerate(sig):
            if i < j:
                continue

            rec.xcorrs[1][i, j] = (sg.correlate(sig1, sig2, 'same')
                                   * (1 / (sig.shape[1] - np.abs(lags))))

    for i in range(sig.shape[0]):
        for j in range(sig.shape[0]):
            if i >= j:
                continue

            rec.xcorrs[1][i, j] = rec.xcorrs[1][j, i]

    rec.data.close()


# for the even more expensive stuff involving time lags, consider epoching
# parallelizing with con.fut.ProcessPoolExecutor
def compute_mutual_info(rec: Recording):
    """
    Compute the mutual information of the data in the Recording object.

    :param rec: The recording object.
    :type rec: Recording
    """
    data = rec.get_data()
    for i, sig1 in enumerate(data):
        for j, sig2 in enumerate(data):
            if i < j:
                continue

            data.mutual_informations[i, j] = mutual_information((sig1, sig2))

    n_channels = len(rec.selected_electrodes)
    for i in range(n_channels):
        for j in range(n_channels):
            if i >= j:
                continue

            rec.mutual_infos[i, j] = data.mutual_infos[j, i]

    rec.data.close()


# PyIF.compute_te already uses numba/gpu
# Parallelizing the loop may be an option depending on the load
# concurrent.futures.ProcessPoolExecutor
def compute_transfer_entropy(rec: Recording, lag_ms: int = 1000):
    """
    Compute the transfer entropy of the data in the Recording object.

    :param rec: The recording object.
    :type rec: Recording

    :param lag_ms: The time lag to use for the transfer entropy computation.
    :type lag_ms: int
    """
    data = rec.get_data()
    n_els = data.shape[0]
    data.transfer_entropies = np.zeros((n_els, n_els))
    lags = int(lag_ms * 0.001 * rec.sampling_rate)
    lags = max(lags, 1)

    for i, sig1 in enumerate(data):
        for j, sig2 in enumerate(data):
            if i == j:
                continue

            rec.transfer_entropies[i, j] = te_compute(sig1,
                                                      sig2,
                                                      embedding=lags)
    rec.data.close()


# parallelize
def compute_coherence(rec: Recording):
    """
    Compute the coherence of the data in the Recording object.

    :param rec: The recording object.
    :type rec: Recording
    """
    data = rec.get_data()
    coherences = []
    for i, sig1 in enumerate(data):
        coherences.append([])
        for j, sig2 in enumerate(data):
            if i < j:
                continue

            coh = ep.spectral.multitaper_coherence(sig1.T, sig2.T,
                                                   fs=rec.sampling_rate)
            coherences[i].append(coh)

    rec.coherences = np.array(coherences)
    rec.data.close()


# parallelize
def compute_granger_causality(rec: Recording, lag_ms=1000):
    """
    Compute the Granger causality of the data in the Recording object.

    :param rec: The recording object.
    :type rec: Recording

    :param lag_ms: The time lag to use for the Granger causality computation.
    :type lag_ms: int
    """
    lags = int(lag_ms * 0.001 * rec.sampling_rate)
    data = rec.get_data()
    cgs = []
    for i, sig1 in enumerate(data):
        cgs.append([])
        for j, sig2 in enumerate(data):
            if i < j:
                continue
            caus = pairwise_granger(np.hstack((sig1.T, sig2.T)),
                                    max_order=lags)
            cgs[i].append(caus)

    rec.granger_causalities = cgs
    rec.data.close()


# parallelize
def compute_spectral_granger(rec: Recording):
    """
    Compute the spectral Granger causality of the data in the Recording object.

    :param rec: The recording object.
    :type rec: Recording
    """
    data = rec.get_data()
    spectral_cgs = []
    for i, sig1 in enumerate(data):
        spectral_cgs.append([])
        for j, sig2 in enumerate(data):
            if i < j:
                continue

            scgs = pairwise_spectral_granger(sig1.T, sig2.T,
                                             fs=rec.sampling_rate)
            spectral_cgs[i].append(scgs)

    rec.spectal_granger = spectral_cgs
    rec.data.close()


def compute_current_source_density(rec: Recording):
    """
    Compute the current source density of the data in the Recording object.

    :param rec: The recording object.
    :type rec: Recording
    """
    data = rec.get_data()
    coords = [[(int(s)-1) * 200 * pq.um for s in name.split() if s.isdigit()]
              for name in rec.names()]

    signal = neo.AnalogSignal(data, units='V',
                              sampling_rate=rec.sampling_rate*pq.Hz)

    rec.csd = ep.current_source_density.estimate_csd(signal,
                                                     coords,
                                                     'KCSD2D')
    rec.data.close()

import antropy as ant
import numba as nb
import numpy as np
import scipy.signal as sg

from model.data import Data


# @nb.njit(parallel=True)
def compute_snrs(data: Data):
    signals = data.data
    data.snrs = np.square(np.mean(signals, axis=-1)) / np.var(signals, axis=-1)


# @nb.njit(parallel=True)
def compute_rms(data: Data) -> np.ndarray:
    data.rms = np.sqrt(np.mean(np.square(data.data), axis=-1))


# @nb.njit(parallel=True)
def compute_derivative(data: Data):
    data.derivatives = np.diff(data.data) * data.sampling_rate # x / 1 /sampling period == x * sampling_period


def compute_mv_avg(data: Data, w: int=None):
    data.mv_means = moving_avg(data.data, w, fs=data.sampling_rate)


# @nb.njit(parallel=True)
def moving_avg(sig: np.ndarray, w: int, fs: int) -> np.ndarray:
    if w is None:
        w = int(np.round(0.01 * fs)) # 10 ms
    if w % 2 == 0:
        w = w + 1

    pad = (w - 1) / 2
    abs_pad = np.pad(np.absolute(sig), (pad, pad), "edge")
    ret = np.cumsum(abs_pad, dtype=float, axis=-1)
    ret[:, w:] = ret[:, w:] - ret[:, :-w]

    return ret[w - 1:] / w


# @nb.njit(parallel=True)
def compute_mv_vars(data: Data, w: int=None):
    sigs = data.data
    sq_dev = np.square((sigs.T - np.mean(sigs, axis=-1)).T)
    data.mv_vars = moving_avg(sq_dev, w=w, fs=data.sampling_rate)


# @nb.njit(parallel=True)
def compute_mv_mads(data: Data, w: int=None):
    sigs = data.data
    abs_devs = np.absolute((sigs.T - np.mean(sigs, axis=-1)).T)
    data.mv_mads = moving_avg(abs_dev, w=w, fs=data.sampling_rate)


# No njit as scipy.signal is not supported & scipy already calls C routines
def compute_envelopes(data: Data):
    data.envelopes = np.absolute(sg.hilbert(data.data))


# antropy uses numba
# Parallelize for loop w. conc.fut.ProcessPoolExec maybe
def compute_entropies(data: Data):
    n_els = data.data.shape[0]
    data.entropies = np.zeros(n_els)
    for i in range(n_els):
        data.entropies[i] = ant.app_entropy(data.data[i])


# @nb.njit
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

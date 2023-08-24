"""
TODO
"""
import antropy as ant
# import numba as nb
import numpy as np

from model.data import Data


# @nb.njit(parallel=True)
def compute_snrs(data: Data):
    signals = data.data
    mean_squared = np.square(np.mean(signals, axis=-1))
    data.channels_df['SNR'] = mean_squared / np.var(signals, axis=-1)


# @nb.njit(parallel=True)
def compute_rms(data: Data) -> np.ndarray:
    data.channels_df['RMS'] = np.sqrt(np.mean(np.square(data.data), axis=-1))


# antropy uses numba
# Parallelize for loop w. conc.fut.ProcessPoolExec maybe
def compute_entropies(data: Data):
    n_els = data.data.shape[0]
    entropies = np.zeros(n_els)
    for i in range(n_els):
        entropies[i] = ant.app_entropy(data.data[i])

    data.channels_df['Apprx_Entropy'] = entropies


# @nb.njit
def bin_amplitude(data: Data, new_sr: int = 500) -> np.ndarray:
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

"""
TODO
"""
import antropy as ant
from numba import njit, prange
import numpy as np

from model.data import Recording


@njit(parallel=True)
def compute_rms_jit(signals: np.ndarray) -> np.ndarray:
    """
    Compute the root mean square (RMS) of the signals using numbas
    just-in-time compiler.

    :param signals: numpy array to calculate RMS value from
    :type signals: np.ndarray

    :return: RMS value of the array
    :rtype: np.ndarray
    """
    return np.sqrt(np.mean(np.square(signals), axis=-1))


@njit(parallel=True)
def compute_snrs_jit(signals: np.ndarray) -> np.ndarray:
    """
    Compute the signal-to-noise ratio (SNR) of the signals using numbas
    just-in-time compiler.

    :param signals: numpy array to calculate SNR value from
    :type signals: np.ndarray

    :return: SNR value of the array
    :rtype: np.ndarray
    """
    mean_squared = np.square(np.mean(signals, axis=-1))
    return mean_squared / np.var(signals, axis=-1)


@njit(parallel=True)
def compute_entropies_jit(data: np.ndarray) -> np.ndarray:
    """
    Compute the approximate entropy of the signals using numbas
    just-in-time compiler for loop parallelization. Antropys app_entropy
    already uses numba under the hood.

    :param data: numpy array to calculate approximate entropy from
    :type data: np.ndarray

    :return: approximate entropy of the array
    :rtype: np.ndarray
    """
    n_els = data.shape[0]
    entropies = np.zeros(n_els)

    for i in prange(data.shape[0]):
        entropies[i] = ant.app_entropy(data[i])

    return entropies


def compute_snrs(rec: Recording):
    """
    Compute SNR of signals in a Recording object and add a new column to the
    channels_df data frame.

    :param rec: Recording object containing signals to be processed
    :type rec: Recording
    """
    rec.channels_df.add_column(['SNR'], [compute_snrs_jit(rec.get_data())])
    rec.data.close()


def compute_rms(rec: Recording):
    """
    Compute RMS of signals in a Recording object and add a new column to the
    channels_df data frame.

    :param rec: Recording object containing signals to be processed
    :type rec: Recording
    """
    rec.channels_df.add_column(['RMS'], [compute_rms_jit(rec.get_data())])
    rec.data.close()


def compute_entropies(rec: Recording):
    """
    Compute entropy values of signals in a Recording object and add a new
    column to the channels_df data frame.

    :param rec: Recording object containing signals to be processed
    :type rec: Recording
    """
    entropies = compute_entropies_jit(rec.get_data())
    rec.channels_df.add_column(['ApproxEntropy'], [entropies])
    rec.data.close()


def bin_amplitude(rec: Recording, new_sr: int = 500) -> np.ndarray:
    """
    Bin the amplitude of the signals in a Recording object to a new sampling
    rate.

    :param rec: Recording object containing signals to be processed
    :type rec: Recording

    :param new_sr: new sampling rate to bin to, defaults to 500
    :type new_sr: int, optional

    :return: binned signals
    :rtype: np.ndarray
    """
    signals = rec.get_data()
    n_bins = new_sr / rec.sampling_rate * signals.shape[1]
    bins = np.zeros((signals.shape[0], n_bins))

    signal_idx = 0
    bin_idx = 0
    while signal_idx < signals.shape[1]:
        t_int = 0
        n_frames_in_bin = 0

        while t_int < 1 / new_sr and signal_idx < signals.shape[1]:
            bins[:, bin_idx] = bins[:, bin_idx] + signals[:, signal_idx]

            t_int = t_int + 1 / rec.sampling_rate
            signal_idx = signal_idx + 1
            n_frames_in_bin = n_frames_in_bin + 1

        bins[:, bin_idx] = bins[:, bin_idx] / n_frames_in_bin
        bin_idx = bin_idx + 1

    rec.data.close()
    return bins

"""
TODO
"""
import scipy.signal as sg
import pdb

from model.data import Recording, SharedArray


# No njit as numpy.fft is not supported & numpy already calls C routines
def compute_psds(rec: Recording):
    """
    Compute the power spectral density of the data in the Recording object.

    :param rec: The recording object.
    :type rec: Recording
    """
    ys = rec.get_data()
    freq, power = sg.welch(ys, fs=rec.sampling_rate, nperseg=256)
    rec.psds = SharedArray(freq), SharedArray(power)


# No njit as scipy.signal is not supported & scipy already calls C routines
def compute_spectrograms(rec: Recording):
    """
    Compute the spectrograms of the data in the Recording object.

    :param rec: The recording object.
    :type rec: Recording
    """
    f, t, sxx = sg.spectrogram(rec.get_data(), fs=rec.sampling_rate,
                               nperseg=256)
    rec.spectrograms = SharedArray(f), SharedArray(t), SharedArray(sxx)

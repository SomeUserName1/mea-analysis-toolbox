"""
TODO
"""
from typing import Optional

import numpy as np
import scipy.signal as sg

from model.data import Recording, SharedArray


def frequency_filter(rec: Recording,
                     stop: bool,
                     low_cut: Optional[float],
                     high_cut: Optional[float],
                     order: Optional[int] = 16):
    """
    A general purpose digital filter for low-pass, high-pass and band-pass
    filtering. Uses the scipy.signal.sosfilt method:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html?highlight=filt%20filt#scipy.signal.sosfilt

    Apply a digital filter forward and backward to a signal.
    This function applies a linear digital filter twice, once forward and once
    backwards. The combined filter has zero phase and a filter order twice that
    of the original.

    :param rec: Input recording object whose signals to filter.
    :type rec: Recording

    :param stop: If True, a bandstop filter is used, therwise a bandpass.
    :type stop: bool

    :param low_cut: Low-pass cutoff frequency in Hz.
    :type low_cut: float

    :param high_cut: High-pass cutoff frequency in Hz.
    :type high_cut: float

    :param order: Order of the filter.
    :type order: int
    """
    fs = rec.sampling_rate
    if low_cut == 0:
        low_cut = None

    if high_cut == fs // 2:
        high_cut = None

# Bandpass or bandstop/notch filter
    if low_cut and high_cut:
        btype = 'bandstop' if stop else 'bandpass'
        sos = sg.butter(N=order, Wn=[low_cut, high_cut], fs=fs,
                        btype=btype, output='sos')
    else:
        # if the stop filter is used invert the limits.
        # i.e. instead of filtering everything below low as in a high pass, we
        # filter everything above low to get a high stop)
        if low_cut:
            # Highpass filter
            cut = high_cut if stop else low_cut
            btype = 'highpass'

        elif high_cut:
            # Lowpass filter
            cut = low_cut if stop else high_cut
            btype = 'lowpass'

        sos = sg.butter(N=order, Wn=cut, btype=btype, fs=fs,
                        output='sos')

    data = rec.get_data()
    data = sg.sosfiltfilt(sos, data)


def downsample(rec: Recording, new_fs: int):
    """
    Downsample the data to a new sampling rate. The new sampling rate must be
    smaller than the current sampling rate.

    :param rec: The recording whichs signals to downsample.
    :type rec: Recording

    :param new_fs: The new sampling rate.
    :type new_fs: int
    """
    q = int(np.round(rec.sampling_rate / new_fs))
    rec.stop_idx = rec.stop_idx / q
    q_it = 0

    prev_data = rec.data
    # determine if we can find a factor that is divisible by 12
    # to downsample the signal without a residual
    for i in range(12):
        if q % (12 - i) == 0:
            q_it = 12 - i
            break

    # if we didnt just use a factor of 10
    if q_it == 0:
        q_it = 10

    downsampled = rec.get_data()
    # As mentioned in the scipy docs, downsampling should be done iteratively
    # if the downsampling factor is larger than 12
    i = 0
    while q > 13:
        # On each iteration we downsample by a factor of q_it
        # and count how often we do that.
        downsampled = sg.decimate(downsampled, q_it)
        q = int(np.round(q / q_it))
        i += 1

    # Adjust the sampling rate with what was downsampled already
    rec.sampling_rate = rec.sampling_rate / q_it**i

    q = int(np.floor(q))
    # if the residual factor is at least 2, downsample by what's left
    if q > 1:
        downsampled = sg.decimate(downsampled, q)
        rec.sampling_rate = int(np.round(rec.sampling_rate / q))

    # if the residual factor is 1, we are done
    # replace the data in the recording object with the downsampled data
    # as it is smaller in size i.e. replace the larger buffer by a smaller one
    rec.data = SharedArray(downsampled)
    # release the larger array from memory
    prev_data.free()


def filter_line_noise(rec: Recording, order: Optional[int] = 16) -> None:
    """
    Filter out the 50 Hz line noise and multiples of it from the data.

    :param rec: The data to filter the line noise from.
    :type rec: Recording

    :param order (int): The order of the filter.
    :type order: int
    """
    freqs = [i * 50 for i in range(1, 10)]

    data = rec.get_data()
    for freq in freqs:
        sos = sg.butter(N=order, Wn=[freq-1.5, freq+1.5], btype='bandstop',
                        output='sos', fs=rec.sampling_rate)
        data = sg.sosfilt(sos, data)

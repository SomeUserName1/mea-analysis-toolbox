from typing import Optional

import numpy as np
import scipy.signal as sg

from model import Data


def frequency_filter(data: Data,
                     filter_type: int,
                     low_cut: Optional[float],
                     high_cut: Optional[float],
                     order: Optional[int] = 16,
                     stop: Optional[bool] = False
                     ) -> np.ndarray:
    """
    A general purpose digital filter for low-pass, high-pass and band-pass
    filtering. Uses the scipy.signal.sosfilt method:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html?highlight=filt%20filt#scipy.signal.sosfilt

    Apply a digital filter forward and backward to a signal.
    This function applies a linear digital filter twice, once forward and once
    backwards. The combined filter has zero phase and a filter order twice that
    of the original.

    Args:
        signal (numpy array): Input signal as a one-dimensional np.array.
        fs (int, float): Sampling frequency of input data.
        filter_type (str): 'BTR' or 'CBY' for Butterworth or Chebychev
            respectively.
        low_cut (int, float): Low-pass cutoff frequency in Hz.
        high_cut (int, float): High-pass cutoff frequency in Hz.
        order (int): Default set to 6.

    Returns:
        y (numpy array (float32)): Return the filtered signal with the same
            shape as the input signal
    """
    fs = data.sampling_rate
    if low_cut == 0:
        low_cut = None

    if high_cut == fs // 2:
        high_cut = None

    # Bandpass or bandstop/notch filter
    if low_cut and high_cut:
        if stop:
            sos = sg.butter(N=order, Wn=[low_cut, high_cut], fs=fs,
                            btype='bandstop', output='sos')
        else:
            sos = sg.butter(N=order, Wn=[low_cut, high_cut],
                            btype='bandpass',  fs=fs,output='sos')

    # if the stop filter is used invert the limits.
    # i.e. instead of filtering everything below low as in a high pass, we
    # filter everything above low to get a high stop)
    if stop:
        temp = low
        low = high
        high = temp

    if low_cut and not high_cut:
        # Highpass filter
        sos = sg.butter(N=order, Wn=low_cut, btype='highpass', fs=fs,
                        output='sos')
    elif high_cut and not low_cut:
        # Lowpass filter
        sos = sg.butter(N=order, Wn=high_cut, btype='lowpass', fs=fs,
                        output='sos')

    data.data = sg.sosfiltfilt(sos, data.data)


def downsample(data: Data, new_fs: int) -> None:
    """
    Downsample the data to a new sampling rate. The new sampling rate must be
    smaller than the current sampling rate.

    Args:
        data (Data): The data to downsample.
        new_fs (int): The new sampling rate.
    """
    q = int(np.round(data.sampling_rate / new_fs))
    q_it = 0

    # determine if we can find a factor that is divisible by 12
    # to downsample the signal without a residual
    for i in range(12):
        if q % (12 - i) == 0:
            q_it = 12 - i
            break

    # if we didnt just use a factor of 10
    if q_it == 0:
        q_it = 10

    # As mentioned in the scipy docs, downsampling should be done iteratively
    # if the downsampling factor is larger than 12
    i = 0
    while q > 13:
        # On each iteration we downsample by a factor of q_it
        # and count how often we do that.
        data.data = sg.decimate(data.data, q_it)
        q = int(np.round(q / q_it))
        i += 1

    # Adjust the sampling rate with what was downsampled already
    data.sampling_rate = data.sampling_rate / q_it**i

    q = int(np.floor(q))
    # if the residual factor is at least 2, downsample by what's left
    if q > 1:
        data.data = sg.decimate(data.data, q)
        data.sampling_rate = int(np.round(data.sampling_rate / q))


def filter_el_humming(data: Data, order: Optional[int] = 16) -> None:
    """
    Filter out the 50 Hz humming noise (and multiples of it) from the data.

    Args:
        data (Data): The data to filter.
        order (int): The order of the filter.
    """
    freqs = [i * 50 for i in range(1, 10)]

    for freq in freqs:
        sos = sg.butter(N=order, Wn=[freq-0.5, freq+0.5], btype='bandstop',
                        output='sos', fs=data.sampling_rate)

        data.data = sg.sosfilt(sos, data.data)


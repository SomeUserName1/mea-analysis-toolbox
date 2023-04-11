from typing import Optional

import numpy as np
import scipy.signal as sg

from model import Data


def frequency_filter(data: Data, filter_type: int, low_cut: Optional[float], high_cut: Optional[float],
                     order=6, ripple=2, stop=False) -> np.ndarray:
    """
    A general purpose digital filter for low-pass, high-pass and band-pass filtering.
        Uses the scipy.signal.sosfilt method:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html?highlight=filt%20filt#scipy.signal.sosfilt
            Apply a digital filter forward and backward to a signal.
            This function applies a linear digital filter twice, once forward and once backwards.
            The combined filter has zero phase and a filter order twice that of the original.

    Args:
        signal (numpy array): Input signal as a one-dimensional np.array.
        fs (int, float): Sampling frequency of input data.
        filter_type (str): 'BTR' or 'CBY' for Butterworth or Chebychev respectively.
        low_cut (int, float): Low-pass cutoff frequency in Hz.
        high_cut (int, float): High-pass cutoff frequency in Hz.
        order (int): Default set to 6.

    Returns:
        y (numpy array (float32)): Return the filtered signal with the same shape as the input signal
    """
    fs = data.sampling_rate
    if low_cut == 0:
        low_cut = None

    if high_cut == fs // 2:
        high_cut = None

    if low_cut and high_cut:
        if stop:
            if filter_type == 0:
                sos = sg.butter(N=order, Wn=[low_cut, high_cut], fs=fs, btype='bandstop', output='sos')
            elif filter_type == 1:
                sos = sg.cheby1(N=order, rp=ripple, Wn=[low_cut, high_cut],  fs=fs,btype='bandstop', output='sos')

        else:
            if filter_type == 0:
                sos = sg.butter(N=order, Wn=[low_cut, high_cut], btype='bandpass',  fs=fs,output='sos')
            elif filter_type == 1:
                sos = sg.cheby1(N=order, rp=ripple, Wn=[low_cut, high_cut], btype='bandpass', fs=fs, output='sos')

    elif low_cut and not high_cut:
        if filter_type == 0:
            sos = sg.butter(N=order, Wn=low_cut, btype='highpass', fs=fs, output='sos')
        elif filter_type == 1:
            sos = sg.cheby1(N=order, rp=ripple, Wn=low_cut, btype='highpass', fs=fs, output='sos')

    elif high_cut and not low_cut:
        if filter_type == 0:
            sos = sg.butter(N=order, Wn=high_cut, btype='lowpass', fs=fs, output='sos')
        elif filter_type == 1:
            sos = sg.cheby1(N=order, rp=ripple, Wn=high_cut, btype='lowpass', fs=fs, output='sos')

    data.data = sg.sosfilt(sos, data.data)


def downsample(data, new_fs):
    q = int(np.round(data.sampling_rate / new_fs))
    q_it = 0

    for i in range(12):
        if q % (12 - i) == 0:
            q_it = 12 - i
            break

    if q_it == 0:
        q_it = 10

    i = 0
    while q > 13:
        q = int(np.round(q / q_it))
        data.data = sg.decimate(data.data, q_it)
        i += 1

    data.sampling_rate = data.sampling_rate / q_it**i

    q = int(np.floor(q))
    if q != 0:
        data.data = sg.decimate(data.data, q)
        data.sampling_rate = data.sampling_rate / q


def filter_el_humming(data, order=10, ripple=2, ftype=0):
    freqs = [i * 50 for i in range(1, 10)]

    for freq in freqs:
        if ftype == 0:
            sos = sg.butter(N=order, Wn=[freq-0.5, freq+0.5], btype='bandstop', \
                    output='sos', fs=data.sampling_rate)
        else:
            sos = sg.cheby1(N=order, rp=ripple, Wn=[freq-0.5, freq+0.5], btype='bandstop', \
                    output='sos', fs=data.sampling_rate)

        data.data = sg.sosfilt(sos, data.data)


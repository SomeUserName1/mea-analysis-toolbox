import time
from multiprocessing import Process

import antropy as ant
import elephant.signal_processing as epsp
from fooof import FOOOF
import numba as nb
import numpy as np
from PyIF.te_compute import te_compute
import scipy.signal as sg

from model.Data import Data
from model.Event import Event
from views.electrode_grid import create_video, \
       plot_value_grid
from views.time_series_plot import plot_in_grid
from views.raster_plot import plot_raster, plot_psth
from views.event_stats import show_events, export_events


@nb.njit(parallel=True)
def compute_snrs(data: Data, result: Result):
    signals = data.selected_data
    result.snrs = np.mean(signals, axis=-1) ** 2 / np.var(signals, axis=-1)


@nb.njit(parallel=True)
def compute_rms(data: Data, result: Result) -> np.ndarray:
    result.rms = np.sqrt(np.mean(np.square(data.selected_data), axis=-1))


@nb.njit(parallel=True)
def compute_derivative(data: Data, result: Result):
    result.derivatives = np.diff(data.selected_data) * sampling_rate # x / 1 /sampling period == x * sampling_period


def compute_mv_avg(data: Data, result: Result, w: int=None):
    result.mv_means = moving_avg(data.selected_data, w, fs=data.sampling_rate)


@nb.njit(parallel=True)
def moving_avg(sig: np.ndarray, w: int=None, fs: int) -> np.ndarray:
    if w is None:
        w = int(np.round(0.01 * fs)) # 10 ms
    if w % 2 == 0
        w = w + 1

    pad = (w - 1) / 2
    abs_pad = np.pad(np.absolute(sig), (pad, pad), "edge")
    ret = np.cumsum(abs_pad, dtype=float, axis=-1)
    ret[:, w:] = ret[:, w:] - ret[:, :-w]

    return ret[w - 1:] / w


@nb.njit(parallel=True)
def compute_mv_var(data: Data, result: Result, w: int=None):
    sigs = data.selected_data
    sq_dev = np.square((sigs.T - np.mean(sigs, axis=-1)).T)

    result.mv_vars = moving_avg(sq_dev, w=w, fs=data.sampling_rate)


@nb.njit(parallel=True)
def compute_mv_mads(data: Data, result: Result, w: int=None):
    sigs = data.selected_data
    abs_devs = np.absolute((sigs.T - np.mean(sigs, axis=-1)).T)

    result.mv_mads = moving_avg(abs_dev, w=w, fs=data.sampling_rate)


# No njit as scipy.signal is not supported & scipy already calls C routines
def compute_envelopes(data: Data, result: Result):
    result.envelopes = np.absolute(sg.hilbert(data.selected_data))


# No njit as numpy.fft is not supported & numpy already calls C routines
def compute_psds(data: Data) -> tuple[np.ndarray, np.ndarray]:
    ys = data.selected_data
    fft = np.fft.rfft(ys)
    power = 2 / np.square(ys.shape[1]) * (fft.real**2 + fft.imag**2)
    freq = np.fft.rfftfreq(ys.shape[1], 1/data.sampling_rate)

    result.psds = freq, power


# No njit as scipy.signal is not supported & scipy already calls C routines
def compute_spectrograms(data: Data, result: Result):
    result.spectrograms =  sg.spectrogram(data.selected_data,
                                          data.sampling_rate, nfft=1024)


# No njit as fooof is unknown to numba
def compute_periodic_aperiodic_decomp(data: Data,
                                      result: Result,
                                      freq_range: tuple[int, int]=(1, 150)
                                      ) -> FOOOFGroup:
    if result.psds is None:
        compute_psds(data.selected_data

    fg = FOOOFGroup()
    fg.fit(result.psds, freq_range, n_jobs=-1)
    result.fooof_group = fg


# will probably not work with numba.
# alternatively parallelize or extract fooof components into matrix and do 
# subtraction as vectrorized (in a separate fn with numba)
# alternatively parallelize and collect.
# prolly IO bound
@nb.jit(parallel=True)
def detrend_fooof(data: Data):
    if result.fooof_group is None:
        fg = compute_periodic_aperiodic_decomp(data) 

    norm_psd = np.empty((len(data.selected_electrodes), 
                         fg.get_fooof(0).power_spectrum))
    for idx in range(len(data.selected_electrodes):
        fooof = fg.get_fooof(idx)

        if fooof.has_model:
            norm_psd[idx] = fooof.power_spectrum - fooof._ap_fit
        else:
            norm_psd[idx] = None

    result.detrended_psds = norm_psd


# PyIF.compute_te already uses numba/gpu
# Parallelizing the loop may be an option depending on the load
# concurrent.futures.ProcessPoolExecutor
@nb.jit(parallel=True)
def compute_transfer_entropy(data: Data):
    n_channels = len(data.selected_electrodes)
    tes = np.zeros(n_channels, n_channels)
    
    for i, sig1 in enumerate(data.selected_data):
        for j, sig2 in enumerate(data.selected_data):
            if i == j:
                tes[i, j] = 0
            elif i < j:
                continue
            else:
                tes[i, j] = compute_te(sig1, sig2)

    for i in range(n_channels):
        for j in range(n_channels):
            if i >= j:
                continue
            else:
                tes[i, j] = tes[j, i]

    result.transfer_entropies = tes

# antropy uses numba
# Parallelize for loop w. conc.fut.ProcessPoolExec maybe
@nb.jit(parallel=True)
def compute_entropies(data: Data, result: Result):
    entropies = np.zeros(len(data.selected_channels))
    for i in range(len(data.selected_channels)):
        entropies[i] = ant.app_entropy(data.selected)

# GENERAL
# baseline how? 
# just not use it? in bursting signals the mean will be very high, so will then be the threshold if 3*std is applied
# just the first n secs? will not work with older recordings maybe
# use baseline recording? loads the whole file into memory and takes a lot of time 
# 
# ---find_peaks---
# seems to detect some peaks multiple times (check with higher resolution viz)
# prominences are not always correct?
# 
# Maybe parallelize using ProcessPoolExec.
@nb.jit(parallel=True)
def detect_peaks(data: Data, result: Result):
    signals = data.selected_data
    mads = np.mean(np.absolute((signals.T - np.means(signals, axis=-1)).T))
    signals = np.absolute(signals)
    result.peaks = []

    for i in range(len(data.selected_electrodes)):
        result.peaks.append(sg.find_peaks(signals[i], threshold=3*mads[i]))


# TODO continue here
def bin_amplitude(data: Data, new_sr: int=600) -> np.ndarray:
    bins = []
    i = data.start_idx
    while i < data.stop_idx:
        binned = np.zeros(data.data[data.selected_electrodes, i].shape)
        t_int = 0
        j = 0
        while t_int < slow_down * 1/fps:
            binned = (binned
                      + np.absolute(data.data[data.selected_electrodes, i]))
            t_int += 1 / data.sampling_rate
            i += 1

            if i >= t_stop_idx:
                break

            j += 1

        binned = binned / j
        bins.append(binned)

    bins = np.array(bins)


def animate_amplitude_grid(data: Data, fps: int, slow_down: float, \
        t_start: int, t_stop: int) -> None:
    """
    Creates an animation that displays the change in amplitude over the
            electrode grid over time

        @param data: The Data object holding the recordings
        @param fps: The frame rate of the animation. A larger frame rate
            causes better temporal resolution, a small frame rate worse,
            i.e. more binning.

    """


    bins = []
    i = data.start_idx
    while i < data.stop_idx:
        binned = np.zeros(data.data[data.selected_electrodes, i].shape)
        t_int = 0
        j = 0
        while t_int < slow_down * 1/fps:
            binned = (binned
                      + np.absolute(data.data[data.selected_electrodes, i]))
            t_int += 1 / data.sampling_rate
            i += 1

            if i >= t_stop_idx:
                break

            j += 1

        binned = binned / j
        bins.append(binned)

    bins = np.array(bins)

    create_video(data, bins, fps)



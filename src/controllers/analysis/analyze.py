import time
from multiprocessing import Process

import numba as nb
import numpy as np
import scipy.signal as sg
from fooof import FOOOF
from PyIF.te_compute import te_compute

from model.Data import Data
from model.Event import Event
from views.electrode_grid import create_video, \
       plot_value_grid
from views.time_series_plot import plot_in_grid
from views.raster_plot import plot_raster, plot_psth
from views.event_stats import show_events, export_events


@nb.njit(parallel=True)
def compute_snrs(data: Data):
    


@nb.njit(parallel=True)
def compute_rms(data: Data) -> np.ndarray:
    return np.sqrt(np.mean(np.power(data.selected_data, 2)))


@nb.njit(parallel=True)
def compute_abs_mv_means(sig: np.ndarray, w: int=None, fs: int) -> np.ndarray:
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
def compute_mv_stds(data: Data, w: int):
    sigs = data.selected_data
    sq_dev = np.square(sigs - np.mean(sigs))
    return np.sqrt(compute_moving_avg(sq_dev, w=w, fs=data.sampling_rate))


@nb.njit(parallel=True)
def compute_mv_mads(data, w):
    sigs = data.selected_data
    abs_devs = np.abs(sigs - np.mean(sigs))
    return compute_moving_avg(abs_dev, w=w)


# No njit as scipy.signal is not supported & scipy already calls C routines
def compute_envelopes(data: Data) -> np.ndarray:
    return np.absolute(sg.hilbert(data.selected_data))


# No njit as numpy.fft is not supported & numpy already calls C routines
def compute_psds(data: Data) -> tuple[np.ndarray, np.ndarray]:
    ys = data.selected_data
    fft = np.fft.rfft(ys)
    power = 2 / np.square(ys.shape[1]) * (fft.real**2 + fft.imag**2)
    freq = np.fft.rfftfreq(ys.shape[1], 1/data.sampling_rate)
    return freq, power


# No njit as scipy.signal is not supported & scipy already calls C routines
def compute_spectrograms(data: Data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return sg.spectrogram(data_row, fs, nfft=1024)

# No njit as fooof is unknown to numba
def compute_periodic_aperiodic_decomp(data: Data, 
                                      freq_range: tuple[int, int]=(1, 150)
                                      ) -> FOOOFGroup:
    fg = FOOOFGroup(aperiodic_mode='knee')
    fg.fit(compute_psds(data), freq_range, freq_range, n_jobs=-1)

    return fg


def detrend_fooof(data: Data):
    fg = compute_periodic_aperiodic_decomp(data) 
   

def compute_isis(spike_idxs, fs):
    isis = []
    for i, (start_idx, end_idx) in enumerate(zip(spike_idxs[0], spike_idxs[1])):
        if i != 0:
            isis.append((start_idx - prev_end_idx) * 1 / fs)

        prev_end_idx = end_idx

    return np.array(isis)


def show_spectrograms(data):
    fs = []
    ts = []
    sxs = []
    for idx in range(data.data.shape[0]):
        f, t, sx = compute_spectrogram(data.data[idx], data.sampling_rate)
        fs.append(f)
        ts.append(t)
        sxs.append(sx)

    fcs = [np.array(fs), np.array(ts), np.array(sxs)]

    proc = Process(target=plot_in_grid, args=('spectrograms', fcs, data))
    proc.start()
    proc.join()


def show_periodic_aperiodic_decomp(data):
    fms = []
    for idx in range(data.data.shape[0]):
        fms.append(compute_periodic_aperiodic_decomp(data.data[idx], data.sampling_rate))

    np.array(fms)

    proc = Process(target=plot_in_grid, args=('periodic_aperiodic', fms, data))
    proc.start()
    proc.join()





def detect_events_moving_dev(data, method, base_std=None, std_factor=1, window=None, export=False, fname=None):
    signal = data.data
    aggs = []
    bursts_longest = []
    bursts_all = []
    events = []
    # take 5% of the signal as window
    window = int(np.round(data.duration_mus / 1000000 / 20 * data.sampling_rate))
    for i in range(len(data.selected_rows)):
        if method == 1:
            aggs.append(compute_mv_std(signal[i], window))
        elif method == 2:
            aggs.append(compute_mv_mad(signal[i], window))
        else:
            raise RuntimeError("Event detection method is not supported")

        if base_std is None:
            std = np.std(signal[i]) * std_factor
        else:
            std = base_std[i] * std_factor

        bursts = find_event_boundaries(aggs[i], std)
        bursts_all.append(bursts)
        event_idxs = extract_longest_burst(bursts)

        if event_idxs[0] is not None:
            event = create_event(data, data.selected_rows[i], event_idxs, std)
            events.append(event)
            bursts_longest.append(event_idxs)
        else:
            bursts_longest.append([])

    data.events = events
    compute_event_delays(data)
   # compute_event_tes(data)
    show_event_psds(data)
    proc = Process(target=plot_in_grid, args=('time_series', signal, data, aggs, None, bursts_all, bursts_longest))
    proc.start()
    proc.join()

    if export:
        export_events(data, fname)

    return show_events(data)





def detect_peaks_amplitude(data, absolute, std_b=None, local_std_factor=2.5, global_std_factor=0.5):
    if absolute:
        signal = np.absolute(data.data)
    else:
        signal = data.data

    peaks = []
    astd = np.std(data.data)

    for i in range(signal.shape[0]):
        if std_b is None:
            std =  np.std(signal[i]) * local_std_factor
        else:
            std = local_std_factor * std_b[i]

        threshold = std + astd * global_std_factor
        peak, param = sg.find_peaks(signal[i], height=threshold)
        height = param['peak_heights']
        peaks.append(np.array([peak, height]))

    proc = Process(target=plot_in_grid, args=('time_series', signal, data, None, peaks))
    proc.start()
    proc.join()

    proc = Process(target=plot_raster, args=(peaks, data))
    proc.start()
    proc.join()

    proc = Process(target=plot_psth, args=(peaks, data))
    proc.start()
    proc.join()


def show_event_psds(data):
    psds = []
    if len(data.events) == 0:
        return

    fst = data.events[0]
    psd_shape = compute_psd(data.data[data.selected_rows.index(fst.electrode_idx), fst.start_idx : fst.end_idx], data.sampling_rate)[0].shape
    for i in range(data.data.shape[0]):
        has_event = False
        for event in data.events:
            if data.selected_rows.index(event.electrode_idx) == i:
                event_signal = data.data[data.selected_rows.index(event.electrode_idx), event.start_idx : event.end_idx]
                psds.append(compute_psd(event_signal, data.sampling_rate))
                has_event = True
                break

        if not has_event:
            psds.append((np.zeros(psd_shape), np.zeros(psd_shape)))

    psds = np.array(psds)

    proc = Process(target=plot_in_grid, args=('psds', psds, data))
    proc.start()
    proc.join()


def find_event_boundaries(signal, threshold, merge=True):
    start_idxs = []
    stop_idxs = []
    i = 0
    burst = False
    while i < signal.shape[0]:
        if not burst and signal[i] >= threshold:
            start_idxs.append(i)
            burst = True
        if burst and signal[i] < threshold:
            stop_idxs.append(i)
            burst = False

        i += 1

    if burst:
        stop_idxs.append(i-1)

    if merge:
        length = len(start_idxs)
        idx = 0
        while idx < length - 1:
            if stop_idxs[idx] + int(signal.shape[0] / 100) >= start_idxs[idx+1]:
                del stop_idxs[idx]
                del start_idxs[idx+1]
                length = len(start_idxs)
            else:
                idx += 1

    return np.array(start_idxs), np.array(stop_idxs)


def extract_longest_burst(burst_times):
    start_idxs, stop_idxs = burst_times
    max_duration = 0
    longest_burst_idx = None

    for i, start_idx in enumerate(start_idxs):
        if max_duration < stop_idxs[i] - start_idx:
            max_duration = stop_idxs[i] - start_idx
            longest_burst_idx = i

    if longest_burst_idx is None:
        return None, None
    else:
        return start_idxs[longest_burst_idx], stop_idxs[longest_burst_idx]


def create_event(data, electrode_idx, event_idxs, threshold):
    if data.sampling_rate < 200:
        raise RuntimeError("Sampling rate must not be lower than 200 Hz for the band decomposition to work!")

    event_signal = data.data[data.selected_rows.index(electrode_idx), event_idxs[0]:event_idxs[1]]
    duration = (event_idxs[1] - event_idxs[0]) * 1 / data.sampling_rate
    spike_idxs = find_event_boundaries(event_signal, threshold, False)
    rms = compute_rms(event_signal)
    max_amplitude = np.amax(np.absolute(event_signal))
    mean_isi = np.mean(compute_isis(spike_idxs, data.sampling_rate))
    freqs, powers = compute_psd(event_signal, 200)

    delta_start = (np.abs(freqs - 0.5)).argmin()
    delta_stop = (np.abs(freqs - 4)).argmin()
    theta_stop = (np.abs(freqs - 8)).argmin()
    alpha_stop = (np.abs(freqs - 13)).argmin()
    beta_stop = (np.abs(freqs - 30)).argmin()
    gamma_stop = (np.abs(freqs - 90)).argmin()

    band_powers = {}
    band_powers['delta'] = np.mean(powers[delta_start:delta_stop])
    band_powers['theta'] = np.mean(powers[delta_stop:theta_stop])
    band_powers['alpha'] = np.mean(powers[theta_stop:alpha_stop])
    band_powers['beta'] = np.mean(powers[alpha_stop:beta_stop])
    band_powers['gamma'] = np.mean(powers[beta_stop:gamma_stop])

    return Event(electrode_idx, event_idxs[0], event_idxs[1], duration, spike_idxs, rms, max_amplitude, mean_isi, band_powers)







def compute_event_delays(data):
    first_idx = None
    for event in data.events:
        if first_idx is None or first_idx > event.start_idx:
            first_idx = event.start_idx

    for event in data.events:
        event.delay = event.start_idx - first_idx


def compute_event_tes(data):
    if data.events[0].delay is None:
        compute_event_delays(data)

    fst_event = None
    for event in data.events:
        if event.delay == 0:
            fst_event = event
            break

    first_signal = data.data[data.selected_rows.index(fst_event.electrode_idx)]
    for event in data.events:
        if event.delay == 0:
            continue

        event_signal = data.data[data.selected_rows.index(event.electrode_idx)]
        event.te = compute_transfer_entropy(first_signal, event_signal)


def compute_transfer_entropy(x, y):
    return te_compute(x, y, k=1, embedding=1)


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
    prev_start_idx = data.start_idx
    prev_stop_idx = data.stop_idx
    if fps is None:
        fps = 60
    else:
        fps = int(fps)

    if slow_down is None:
        slow_down = 0.1
    else:
        slow_down = float(slow_down)

    if t_start is None:
        start_idx = data.start_idx
    else:
        start_mus = str_to_mus(t_start)

    if t_stop is None:
        stop_idx = data.stop_idx
    else:
        stop_mus = str_to_mus(t_stop)

    data.set_time_window(start_mus, stop_mus)

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

    data.set_time_window(prev_start_mus, prev_stop_mus)




# FIXME sth is off here.
# FIXME just create a normal data object, and the code from analyze!
def extract_baseline_std(baseline, que):
    """
    Extracts the standard deviation per channel from a file without creating a \
            Data object.

        @param baseline: path to the McS h5 file with the baseline data.
        @param selected_rows: the rows of the selected electrodes of the \
                stimulus data (see detect_peaks/detect_events in controllers/ \
                analyze.py)
        @param moving_avg: boolean indicating if the std of the moving \
                average shall be extracted.

        @return the std of the channel or of the moving average of the channel
    """
    file_contents = McsPy.McsData.RawData(baseline)
    stream = file_contents.recordings[0].analog_streams[0]
    num_electrodes = stream.channel_data.shape[0]
    sampling_rate = stream.channel_infos[2].sampling_frequency.magnitude
    data = stream.channel_data
    data = np.array(data)
    # downsample signal to speed things up
    q = int(np.round(sampling_rate / 1000))
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
        data = sg.decimate(data, q_it)
        i += 1

    sampling_rate = sampling_rate / q_it ** i

    q = int(np.floor(q))
    if q != 0:
        data = sg.decimate(data, q)
        sampling_rate = sampling_rate / q
    stds = []
    mv_std_stds = []
    mv_mad_stds = []
    window = int(np.round(data.shape[1] / 20))

    for i in range(num_electrodes):
        abs_dev = np.abs(data[i] - np.mean(data[i]))
        mv_mad = np.convolve(abs_dev, np.ones(window), 'same') / window

        mv_std = np.convolve(np.square(abs_dev), np.ones(window), 'same') / window
        mv_std = np.sqrt(mv_std)

        mv_std_stds.append(np.std(mv_std))
        mv_mad_stds.append(np.std(mv_mad))
        stds.append(np.std(data[i]))

    del file_contents

    que.put((stds, mv_std_stds, mv_mad_stds))


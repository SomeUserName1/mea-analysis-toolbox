# import numba as nb
import numpy as np
import pandas as pd
import scipy.signal as sg

from model.data import Data


# @nb.njit(parallel=True)
def compute_derivatives(data: Data):
    # x / 1 /sampling period == x * sampling_period
    data.derivatives = np.diff(data.data) * data.sampling_rate


def compute_mv_avgs(data: Data, w: int = None):
    data.mv_means = moving_avg(data.data, w, fs=data.sampling_rate)


# @nb.njit(parallel=True)
def moving_avg(sig: np.ndarray, w: int, fs: int) -> np.ndarray:
    if w is None:
        w = int(np.round(0.005 * fs))  # 5 ms
    if w % 2 == 0:
        w = w + 1

    pad = int((w - 1) / 2)
    abs_pad = np.pad(np.absolute(sig), ((0, 0), (pad, pad)), "edge")
    ret = np.cumsum(abs_pad, dtype=float, axis=-1)
    ret[:, w:] = ret[:, w:] - ret[:, :-w]

    return ret[:, w - 1:] / w


# @nb.njit(parallel=True)
def compute_mv_mads(data: Data, w: int = None):
    sigs = data.data
    abs_dev = np.absolute((sigs.T - np.median(sigs, axis=-1)).T)
    data.mv_mads = moving_avg(abs_dev, w=w, fs=data.sampling_rate)


# TODO finish envelope
def compute_envelope_idxs(data, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the
        input signal is too big
    split: bool, optional, if True, split the signal in half along its mean,
        might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """
    s = data.data
    # locals min
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1

    if split:
        # s_mid is zero if s centered around the mean of signal
        s_mid = np.mean(s)
        # pre-sorting of locals min based on position with respect to s_mid
        lmin = lmin[s[lmin] < s_mid]
        # pre-sorting of local max based on position with respect to s_mid
        lmax = lmax[s[lmax] > s_mid]

    # global min of dmin-chunks of locals min
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]])
                for i in range(0, len(lmin), dmin)]]
    # global max of dmax-chunks of locals max
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]])
                 for i in range(0, len(lmax), dmax)]]

    data.envelopes = lmin, lmax

# TODO tomorrow
# IMPL https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data/22640362#22640362
def detect_peaks_dispersion():
    #     # Let y be a vector of timeseries data of at least length lag+2
    # # Let mean() be a function that calculates the mean
    # # Let std() be a function that calculates the standard deviaton
    # # Let absolute() be the absolute value function
    # 
    # # Settings (these are examples: choose what is best for your data!)
    # set lag to 5;          # average and std. are based on past 5 observations
    # set threshold to 3.5;  # signal when data point is 3.5 std. away from average
    # set influence to 0.5;  # between 0 (no influence) and 1 (full influence)
    # 
    # # Initialize variables
    # set signals to vector 0,...,0 of length of y;   # Initialize signal results
    # set filteredY to y(1),...,y(lag)                # Initialize filtered series
    # set avgFilter to null;                          # Initialize average filter
    # set stdFilter to null;                          # Initialize std. filter
    # set avgFilter(lag) to mean(y(1),...,y(lag));    # Initialize first value average
    # set stdFilter(lag) to std(y(1),...,y(lag));     # Initialize first value std.
    # 
    # for i=lag+1,...,t do
    #   if absolute(y(i) - avgFilter(i-1)) > threshold*stdFilter(i-1) then
    #     if y(i) > avgFilter(i-1) then
    #       set signals(i) to +1;                     # Positive signal
    #     else
    #       set signals(i) to -1;                     # Negative signal
    #     end
    #     set filteredY(i) to influence*y(i) + (1-influence)*filteredY(i-1);
    #   else
    #     set signals(i) to 0;                        # No signal
    #     set filteredY(i) to y(i);
    #   end
    #   set avgFilter(i) to mean(filteredY(i-lag+1),...,filteredY(i));
    #   set stdFilter(i) to std(filteredY(i-lag+1),...,filteredY(i));
    # end


# Maybe parallelize using ProcessPoolExec.
# @nb.jit(parallel=True)
def detect_peaks(data: Data, threshold_factor=6):
    if data.mv_mads is None:
        win = int(np.round(0.15 * data.sampling_rate))
        compute_mv_avgs(data, win)

    signals = data.data
    mads = np.median(np.absolute(signals.T - np.median(signals, axis=-1)).T,
                     axis=-1)
    signals = np.abs(signals)
    n_peaks = np.zeros(data.data.shape[0])
    peaks_freq = np.zeros(data.data.shape[0])
    names = data.get_sel_names()
    data.peaks_df = pd.DataFrame([],
                                 columns=["Channel", "PeakIndex", "TimeStamp",
                                          "Amplitude", "InterPeakInterval"])
    rows = [data.peaks_df]
    for i in range(data.data.shape[0]):
        peaks = []
        peak_durations = []
        peak_up_slopes = []
        peak_down_slopes = []

        thresh = threshold_factor*mads[i]

        above_thresh = (signals[i] > thresh).astype(int)
        above_thresh = np.concatenate(([0], above_thresh, [0]))
        abs_diff = np.abs(np.diff(above_thresh))
        above_thresh_idxs = np.where(abs_diff == 1)[0].reshape(-1, 2)
        # TODO use envelope 
        for idxs in above_thresh_idxs:
            peak = np.argmax(signals[i][idxs[0]:idxs[1]]) + idxs[0]
            duration = (idxs[1] - idxs[0]) / data.sampling_rate
            up_dx = (peak - idxs[0]) / data.sampling_rate
            up_dy = data.data[i][peak] - data.data[i][idxs[0]]
            down_dx = (idxs[1] - peak) / data.sampling_rate
            down_dy = data.data[i][idxs[0]] - data.data[i][peak]

            peaks.append(peak)
            peak_durations.append(duration)
            peak_up_slopes.append(up_dy / up_dx)
            peak_down_slopes.append(down_dy / down_dx)

        peaks = np.array(peaks).astype(int)
        peak_durations = np.array(peak_durations)

        n_peaks[i] = len(peaks)
        peaks_freq[i] = n_peaks[i] / data.duration_mus / 1000000

        peak_ampls = data.data[i][peaks]
        channel = np.repeat(names[i], len(peaks))
        peak_times = peaks / data.sampling_rate

        ipi = np.diff(peaks) / data.sampling_rate
        if peaks.shape[0] > 0:
            ipi = np.insert(ipi, 0, np.nan, axis=-1)

        channel_peaks = pd.DataFrame(
                {"Channel": channel,
                 "PeakIndex": peaks,
                 "TimeStamp": peak_times,
                 "Amplitude": peak_ampls,
                 "Duration": peak_durations,
                 "UpSlope": peak_up_slopes,
                 "DownSlope": peak_down_slopes,
                 "InterPeakInterval": ipi}
                )
        rows.append(channel_peaks)

    data.peaks_df = pd.concat(rows)
    data.channels_df['n_peaks'] = n_peaks
    data.channels_df['peak_freq'] = peaks_freq


# GENERAL
# baseline how?
# just not use it? in bursting signals the mean will be very high, so will then be the threshold if 3*std is applied
# just the first n secs? will not work with older recordings maybe
# use baseline recording? loads the whole file into memory and takes a lot of time
#
# RN used is the signal + 3 *the MAD of the signal. Alternative: use roling mad instead of signal
#
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


def compute_isis(spike_idxs, fs):
    isis = []
    for i, (start_idx, end_idx) in enumerate(zip(spike_idxs[0], spike_idxs[1])):
        if i != 0:
            isis.append((start_idx - prev_end_idx) * 1 / fs)

        prev_end_idx = end_idx

    return np.array(isis)


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



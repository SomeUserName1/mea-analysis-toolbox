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
    abs_pad = np.pad(sig, ((0, 0), (pad, pad)), "reflect")
    ret = np.cumsum(abs_pad, dtype=float, axis=-1)
    ret[:, w:] = ret[:, w:] - ret[:, :-w]

    return ret[:, w - 1:] / w


# @nb.njit(parallel=True)
def compute_mv_mads(data: Data, w: int = None):
    sigs = data.data
    compute_mv_avgs(data, w)
    abs_dev = np.absolute((sigs.T - np.mean(sigs, axis=-1)).T)
    data.mv_mads = moving_avg(abs_dev, w=w, fs=data.sampling_rate)


def compute_envelopes(data, win=1):
    data.envelopes = envelopes(data.data, win)


def envelopes(s, win):
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
    # locals min
    lmin = (np.diff(np.sign(np.diff(s))) > 0)
    lmin = [lmin[i].nonzero()[0] + 1 for i in range(lmin.shape[0])]

    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0)
    lmax = [lmax[i].nonzero()[0] + 1 for i in range(lmax.shape[0])]

    # global min of win-chunks of locals min
    lmin = [np.array([lmin[j][i + np.argmin(s[j][lmin[j][i:i + win]])]
            for i in range(0, lmin[j].shape[0], win)])
            for j in range(len(lmin))]
    # global max of win-chunks of locals max
    lmax = [np.array([lmax[j][i + np.argmax(s[j][lmax[j][i:i + win]])]
            for i in range(0, lmax[j].shape[0], win)])
            for j in range(len(lmax))]

    return lmin, lmax


def detect_peaks(data: Data,
                 mad_win: float = None,
                 env_win: float = None,
                 env_percentile: int = None,
                 mad_thrsh: float = None,
                 env_thrsh: float = None):
    detect_peaks_mv_mad_envs_thresh(data, mad_win, env_win, env_percentile,
                                    env_thrsh)


# Maybe parallelize using ProcessPoolExec.
# @nb.jit(parallel=True)
def detect_peaks_mv_mad_envs_thresh(data: Data,
                                    mad_win: float = 0.05,
                                    env_win: float = 0.1,
                                    env_percentile: int = 5,
                                    mad_thrsh: float = 1.5,
                                    env_thrsh: float = 2):
    if mad_win is None:
        mad_win = 0.05
    if env_win is None:
        env_win = 0.1
    if env_percentile is None:
        env_percentile = 5
    if mad_thrsh is None:
        mad_thrsh = 1.5
    if env_thrsh is None:
        env_thrsh = 2
    win = int(np.round(mad_win * data.sampling_rate))
    compute_mv_mads(data, win)

    win = int(np.round(env_win * data.sampling_rate))
    _, mad_env = envelopes(data.mv_mads, win)
    data.mad_env = mad_env

    compute_envelopes(data, win)

    signals = data.mv_mads

    n_peaks = np.zeros(data.data.shape[0])
    peaks_freq = np.zeros(data.data.shape[0])
    names = data.get_sel_names()
    data.peaks_df = pd.DataFrame([],
                                 columns=["Channel", "PeakIndex", "TimeStamp",
                                          "Amplitude", "InterPeakInterval"])
    data.lower = np.zeros(data.data.shape[0])
    data.upper = np.zeros(data.data.shape[0])
    data.mad_thresh = np.zeros(data.data.shape[0])
    rows = [data.peaks_df]
    for i in range(data.data.shape[0]):
        print(f"channel {i}")
        peaks = []
        peak_durations = []
        peak_left_slopes = []
        peak_right_slopes = []

        data.mad_thresh[i] = (mad_thrsh * np.percentile(
                                data.mv_mads[i][mad_env[i]], env_percentile))
        min_env = data.envelopes[0][i]
        max_env = data.envelopes[1][i]
        lower_thresh = env_thrsh * np.percentile(data.data[i][min_env],
                                                 (100 - env_percentile))
        data.lower[i] = lower_thresh
        upper_thresh = env_thrsh * np.percentile(data.data[i][max_env],
                                                 env_percentile)
        data.upper[i] = upper_thresh

        # we have a peak, if the mad is above the respective threshold and the
        # sign is according to that threshold
        above_thresh = (signals[i] > data.mad_thresh[i]).astype(int)

        above_thresh = np.concatenate(([0], above_thresh, [0]))
        abs_diff = np.abs(np.diff(above_thresh))
        above_thresh_idxs = np.where(abs_diff == 1)[0].reshape(-1, 2)
        # TODO we assume that the signal will cross the threshold left to the
        # mad signals crossing for the lower bound and right to the upperbound
        for idxs in above_thresh_idxs:
            print(f"indxes {idxs}")
            idxs_peaks = []
            if any(data.data[i][idxs[0]:idxs[1]] > upper_thresh):
                # find actual boundaries, as the current ones are based on a
                # a moving quantity with relatively large window size
                #    for t in range(idxs[0], 0):
                #        if data.data[i][t] < upper_thresh:
                #            idxs[0] = t
                #            break
                #    for t in range(idxs[1], 1):
                #        if data.data[i][t] < upper_thresh:
                #            idxs[1] = t
                #            break
                idxs_peaks.append(np.argmax(data.data[i][idxs[0]:idxs[1]])
                                  + idxs[0])

            if any(data.data[i][idxs[0]:idxs[1]] < lower_thresh):
                #    for t in range(idxs[0], 0):
                #        if data.data[i][t] > lower_thresh:
                #            idxs[0] = t
                #            break
                #    for t in range(idxs[1], 1):
                #        if data.data[i][t] > lower_thresh:
                #            idxs[1] = t
                #            break
                idxs_peaks.append(np.argmin(data.data[i][idxs[0]:idxs[1]])
                                  + idxs[0])

            for peak in idxs_peaks:
                duration = (idxs[1] - idxs[0]) / data.sampling_rate
                l_dx = (peak - idxs[0]) / data.sampling_rate
                l_dx = l_dx if l_dx != 0 else 1 / data.sampling_rate
                l_dy = data.data[i][peak] - data.data[i][idxs[0]]
                l_dy = (l_dy if l_dy != 0
                        else data.data[i][peak] - data.data[i][idxs[0] - 1])

                r_dx = (idxs[1] - peak) / data.sampling_rate
                r_dx = r_dx if r_dx != 0 else 1 / data.sampling_rate
                r_dy = data.data[i][idxs[0]] - data.data[i][peak]
                r_dy = (r_dy if r_dy != 0
                        else data.data[i][peak] - data.data[i][idxs[0] - 1])

                peaks.append(peak)
                peak_durations.append(duration)
                peak_left_slopes.append(l_dy / l_dx)
                peak_right_slopes.append(r_dy / r_dx)

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
                 "LeftSlope": peak_left_slopes,
                 "RightSlope": peak_right_slopes,
                 "InterPeakInterval": ipi}
                )
        rows.append(channel_peaks)

    data.peaks_df = pd.concat(rows)
    data.peaks_df.sort_values(by=["PeakIndex"])
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



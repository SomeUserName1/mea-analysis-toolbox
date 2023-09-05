import numpy as np
import pandas as pd
import scipy.signal as sg

from model.data import Recording, SharedArray, SharedDataFrame


def compute_derivatives_jit(data: np.ndarray, fs: int) -> np.ndarray:
    """
    Compute the first derivative of the signals using numbas
    just-in-time compiler.

    :param data: numpy array to calculate the first derivative from
    :type data: np.ndarray

    :return: first derivative of the array
    :rtype: np.ndarray
    """
    return np.diff(data) * fs


def moving_avg(sig: np.ndarray, w: int, fs: int = None) -> np.ndarray:
    """
    Compute the moving average of the signals.

    :param sig: numpy array to calculate the moving average from
    :type sig: np.ndarray

    :param w: window size
    :type w: int

    :return: moving average of the array
    :rtype: np.ndarray
    """
    if w is None:
        assert fs is not None, ("Either window size or sampling rate"
                                " must be given")
        w = int(np.round(0.005 * fs))  # 5 ms
    if w % 2 == 0:
        w = w + 1

    pad = int((w - 1) / 2)
    padded = np.pad(sig, ((0, 0), (pad, pad)), "reflect")
    ret = np.cumsum(padded, axis=-1)
    ret[:, w:] = ret[:, w:] - ret[:, :-w]

    return ret[:, w - 1:] / w


# No numba as modifying lists in the loop is not supported
def envelopes(s: np.ndarray,
              win: int
              ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Compute the envelopes of the signals using numbas
    just-in-time compiler.
    First detect local minima and maxima and then compute the global minima
    and maxima of the windows around the local minima and maxima.

    :param s: numpy array to calculate the envelopes from
    :type s: np.ndarray

    :param win: window size for the global minima and maxima search in terms of
        indexes
    :type win: int

    :return: envelopes of the array
    :rtype: tuple[list[np.ndarray], list[np.ndarray]]
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


def compute_derivatives(rec: Recording):
    """
    Compute the first derivative of the signals.

    :param rec: the recording object
    :type rec: Recording
    """
    data = rec.get_data()
    rec.derivatives = SharedArray(
            compute_derivatives_jit(data, rec.sampling_rate)
                                 )
#    rec.data.close()


def compute_mv_avgs(rec: Recording, w: int = None):
    """
    Compute the moving average.
    Pad the signal by window size / 2 on both sides to get the same shape
    as the input array.

    :param sig: numpy array to calculate the moving average from
    :type sig: np.ndarray

    :param w: window size
    :type w: int

    :return: moving average of the array
    :rtype: np.ndarray
    """
    data = rec.get_data()
    rec.mv_avgs = SharedArray(moving_avg(data, w))
#    rec.data.close()


def compute_mv_mads(rec: Recording, w: int = None):
    """
    Compute the moving mean absolute deviation of the signals using numbas
    just-in-time compiler.
    The here computed quantity first subtracts the mean of the whole signal
    and computes a moving average of the absolute deviation from that mean.
    To emphasize, the calculation of the mean is not within the window so
    strictly speaking this is not exactly moving mean absolute deviation.

    :param data: numpy array to calculate the moving MAD from
    :type data: np.ndarray

    :param w: window size
    :type w: int

    :param fs: sampling rate
    :type fs: int

    :return: moving MAD of the array
    :rtype: np.ndarray
    """
    sigs = rec.get_data()
    abs_dev = np.absolute(sigs.T - np.mean(sigs, axis=-1)).T
    rec.mv_mads = SharedArray(moving_avg(abs_dev, w))
#    rec.data.close()


def compute_envelopes(rec: Recording, win: int = 100):
    """
    Compute the envelopes of the signals.

    :param rec: the recording object
    :type rec: Recording
    """
    data = rec.get_data()
    # tuple of lists containing ndarrays.
    #  shared lists do not support ndarrays.
    # Also with a reasonable window size, the lists should not be too large
    # e.g. for 0.1s window size, 1kHz sampling rate, and a duration of 120 s
    # the lists will contain 1200 elements * number of selected channels.
    # For 10 selected channels, this is 12000 elements * 4 bytes = 48 kB.
    rec.envelopes = envelopes(data, win)
#    rec.data.close()


def detect_peaks(rec: Recording,
                 mad_win: float = None,
                 env_win: float = None,
                 env_percentile: int = None,
                 mad_thrsh_f: float = None,
                 env_thrsh_f: float = None):
    """
    Detect peaks in the signals of a recording object.
    The detection is based on the moving MAD of the signals and the envelope
    of the moving MAD and amplitude thresholds. The moving MAD is used to
    detect peaks as it is smoother than the signal itself and increases
    strongly when the siginal is peaking or bursting. The envelopes are used
    to estimate the noise noise levels per channel. The thresholds are
    computed as a factor times a percentile of the respective envelope.
    The factor is given by the user, as well as the percentile.

    :param rec: the recording object
    :type rec: Recording

    :param mad_win: window size for the moving MAD, defaults to 0.05
    :type mad_win: float, optional

    :param env_win: window size for the envelopes, defaults to 0.1
    :type env_win: float, optional

    :param env_percentile: percentile of the envelope to use as threshold,
        defaults to 5
    :type env_percentile: int, optional

    :param mad_thrsh_f: factor to multiply the percentile of the MAD envelope
        to use as threshold, defaults to 1.5
    :type mad_thrsh_f: float, optional

    :param env_thrsh_f: factor to multiply the percentile of the signal
        envelope to use as threshold, defaults to 2
    :type env_thrsh_f: float, optional
    """
    if mad_win is None:
        mad_win = 0.05
    if env_win is None:
        env_win = 0.1
    if env_percentile is None:
        env_percentile = 5
    if mad_thrsh_f is None:
        mad_thrsh_f = 1.5
    if env_thrsh_f is None:
        env_thrsh_f = 2

    fs = rec.sampling_rate
    names = rec.get_sel_names()

    # Compute moving mean absolute deviation of the signals, used to detect
    # peaks as the moving MAD is smoother than the signal itself and increases
    # strongly when the siginal is peaking or bursting.
    win = int(np.round(mad_win * fs))
    compute_mv_mads(rec, win)

    # compute the envelope of the MAD to estimate the noise threshold of
    # the moving MAD signal. Attach it to the recording object to be able to
    # plot it later, when tuning the parameters.
    win = int(np.round(env_win * fs))
    mv_mads = rec.mv_mads.read()
    _, mad_env = envelopes(mv_mads, win)
    rec.mad_env = mad_env

    # compute the envelope of the sigal to later estimate the noise levels
    # per channel
    compute_envelopes(rec, win)

    data = rec.get_data()
    rows, n_peaks, peaks_freq, lower, upper, mad_thresh = (
            detect_peaks_mv_mad_envs_thresh(data,
                                            rec.sampling_rate,
                                            names,
                                            mv_mads,
                                            mad_env,
                                            rec.envelopes,
                                            env_percentile,
                                            mad_thrsh_f,
                                            env_thrsh_f)
            )

    rec.lower = lower
    rec.upper = upper
    rec.mad_thresh = mad_thresh

#    rec.data.close()
#    rec.mv_mads.close()

    # concatenate the list of data frames into one data frame,
    # sort it by channel and peak index and attach it to the recording object
    peaks_df = pd.concat(rows)
    peaks_df.sort_values(by=["Channel", "PeakIndex"], inplace=True)
    rec.peaks_df = SharedDataFrame(peaks_df)
    # add the number of peaks and the peak frequency to the channels data frame
    rec.channels_df.add_cols(['n_peaks', 'peak_freq'], [n_peaks, peaks_freq])


# Maybe parallelize using ProcessPoolExec.
# @njit(parallel=True)
def detect_peaks_mv_mad_envs_thresh(data: np.ndarray,
                                    fs: int,
                                    names: list[str],
                                    mv_mads: np.ndarray,
                                    mad_env: list[np.ndarray],
                                    envs: tuple[list[np.ndarray],
                                                list[np.ndarray]],
                                    env_percentile: int = 5,
                                    mad_thrsh_f: float = 1.5,
                                    env_thrsh_f: float = 2):

    n_peaks = np.zeros(data.shape[0])
    peaks_freq = np.zeros(data.shape[0])

    lower = np.zeros(data.shape[0])
    upper = np.zeros(data.shape[0])
    mad_thresh = np.zeros(data.shape[0])
    # we'll write concurrently to the list and sort it afterwards
    rows = []
    for i in range(data.shape[0]):  # prange
        peaks = []
        peak_durations = []
        peak_left_slopes = []
        peak_right_slopes = []

        # the theshold for a peak/burst is a factor times a percentile of
        # the MAD signal envelope. The factor is given by the user, as well as
        # the percentile. The envelope percentile approach is chosen to detect
        # the noise level as the signal is very noisy and so is the moving MAD.
        mad_thresh[i] = (mad_thrsh_f * np.percentile(
                                mv_mads[i][mad_env[i]], env_percentile))

        # Analogously, the thrshold for the signal amplitudes is based on a
        # percentile of the envelope of the signal itself with large window
        # (0.1s for example) multiplied by a facor. The factor is given by the
        # user, as well as the percentile.
        min_env = envs[0][i]
        max_env = envs[1][i]
        lower[i] = env_thrsh_f * np.percentile(data[i][min_env],
                                               (100 - env_percentile))
        upper[i] = env_thrsh_f * np.percentile(data[i][max_env],
                                               env_percentile)

        # we have a peak/burst, if the mad is above the respective threshold
        above_thresh = (mv_mads[i] > mad_thresh[i]).astype(int)

        # find the indices where the signal crosses the threshold
        above_thresh = np.concatenate(([0], above_thresh, [0]))
        abs_diff = np.abs(np.diff(above_thresh))
        # and convert it to an array of tuples of the form (start, stop)
        above_thresh_idxs = np.where(abs_diff == 1)[0].reshape(-1, 2)
        # TODO we assume that the signal will cross the threshold left to the
        # mad signals crossing for the lower bound and right to the upperbound
        for (start, stop) in above_thresh_idxs:
            idxs_peaks = []
            if any(data[i][start:stop] > upper[i]):
                # find actual boundaries, as the current ones are based on a
                # a moving quantity with relatively large window size
                #    for t in range(start, 0):
                #        if data.data[i][t] < upper_thresh:
                #            start = t
                #            break
                #    for t in range(stop, 1):
                #        if data.data[i][t] < upper_thresh:
                #            stop = t
                #            break
                idxs_peaks.append(np.argmax(data[i][start:stop])
                                  + start)

            if any(data[i][start:stop] < lower[i]):
                #    for t in range(start, 0):
                #        if data.data[i][t] > lower_thresh:
                #            start = t
                #            break
                #    for t in range(stop, 1):
                #        if data.data[i][t] > lower_thresh:
                #            stop = t
                #            break
                idxs_peaks.append(np.argmin(data[i][start:stop])
                                  + start)

            for peak in idxs_peaks:
                duration = (stop - start) / fs
                l_dx = (peak - start) / fs
                l_dx = l_dx if l_dx != 0 else 1 / fs
                l_dy = data[i][peak] - data[i][start]
                l_dy = (l_dy if l_dy != 0
                        else data[i][peak] - data[i][start - 1])

                r_dx = (stop - peak) / fs
                r_dx = r_dx if r_dx != 0 else 1 / fs
                r_dy = data[i][start] - data[i][peak]
                r_dy = (r_dy if r_dy != 0
                        else data[i][peak] - data[i][start - 1])

                peaks.append(peak)
                peak_durations.append(duration)
                peak_left_slopes.append(l_dy / l_dx)
                peak_right_slopes.append(r_dy / r_dx)

        peaks = np.array(peaks).astype(int)
        peak_durations = np.array(peak_durations)

        n_peaks[i] = len(peaks)
        peaks_freq[i] = n_peaks[i] / fs / 1000000

        peak_ampls = data[i][peaks]
        channel = np.repeat(names[i], len(peaks))
        peak_times = peaks / fs

        ipi = np.diff(peaks) / fs
        if peaks.shape[0] > 0:
            ipi = np.hstack((np.array([np.nan]), ipi))

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

    return rows, n_peaks, peaks_freq, lower, upper, mad_thresh


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



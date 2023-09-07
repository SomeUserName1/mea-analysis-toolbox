import numpy as np
import pandas as pd
import pdb

from model.data import Recording, SharedArray


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

    # concatenate the list of data frames into one data frame,
    # sort it by channel and peak index and attach it to the recording object
    rec.peaks_df = pd.concat(rows)
    rec.peaks_df.sort_values(by=["Channel", "PeakIndex"], inplace=True)
    rec.channels_df['n_peaks'] = n_peaks
    rec.channels_df['peak_freq'] = peaks_freq


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
        starts = []
        stops = []

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
        for (start, stop) in above_thresh_idxs:
            if any(data[i][start:stop] > upper[i]):
                peaks.append(np.argmax(data[i][start:stop]) + start)
                # find actual boundaries, as the current ones are based on a
                # a moving quantity with relatively large window size
                p_start = start
                p_stop = stop
                for t in range(peaks[-1], 0):
                    if data[i][t] < upper[i]:
                        p_start = t
                        break
                for t in range(peaks[-1], data.shape[1]):
                    if data[i][t] < upper[i]:
                        p_stop = t
                        break

                peak_durations.append((p_stop - p_start) / fs)
                starts.append(p_start)
                stops.append(p_stop)

            if any(data[i][start:stop] < lower[i]):
                peaks.append(np.argmin(data[i][start:stop]) + start)

                p_start = start
                p_stop = stop
                for t in range(peaks[-1], 0):
                    if data[i][t] > lower[i]:
                        p_start = t
                        break
                for t in range(peaks[-1], data.shape[1]):
                    if data[i][t] < lower[i]:
                        p_stop = t
                        break

                peak_durations.append((p_stop - p_start) / fs)
                starts.append(p_start)
                stops.append(p_stop)

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
                 "StartIndex": starts,
                 "StopIndex": stops,
                 "Duration": peak_durations,
                 "InterPeakInterval": ipi}
                )
        rows.append(channel_peaks)

    return rows, n_peaks, peaks_freq, lower, upper, mad_thresh


def detect_events(rec: Recording,
                  mad_win: float = None,
                  env_percentile: int = None,
                  mad_thrsh_f: float = None):
    """
    Detect events in the signals of a recording object.
    The detection is based on the moving MAD of the signals and the envelope
    of the moving MAD. The moving MAD is used to detect events as it is
    smoother than the signal itself and increases strongly when the siginal is
    bursting. The envelopes are used to estimate the noise noise levels per
    channel. The thresholds are computed as a factor times a percentile of the
    envelope of the moving mad. The factor is given by the user, as well as
    the percentile.

    :param rec: the recording object
    :type rec: Recording

    :param mad_win: window size for the moving MAD, defaults to 0.05
    :type mad_win: float, optional

    :param env_percentile: percentile of the envelope to use as threshold,
        defaults to 5
    :type env_percentile: int, optional

    :param mad_thrsh_f: factor to multiply the percentile of the MAD envelope
        to use as threshold, defaults to 1.5
    :type mad_thrsh_f: float, optional
    """
    if mad_win is None:
        mad_win = 0.05
    if env_percentile is None:
        env_percentile = 5
    if mad_thrsh_f is None:
        mad_thrsh_f = 1.5

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
    mv_mads = rec.mv_mads.read()
    _, mad_env = envelopes(mv_mads, win)
    rec.mad_env = mad_env

    data = rec.get_data()
    rows, n_peaks, peaks_freq, lower, upper, mad_thresh = (
            detect_peaks_mv_mad_envs_thresh(data,
                                            rec.sampling_rate,
                                            names,
                                            mv_mads,
                                            mad_env,
                                            env_percentile,
                                            mad_thrsh_f)
            )

    rec.event_mad_thresh = mad_thresh

    # concatenate the list of data frames into one data frame,
    # sort it by channel and peak index and attach it to the recording object
    rec.events_df = pd.concat(rows)

    extract_event_measures(rec)


# Maybe parallelize using ProcessPoolExec.
# @njit(parallel=True)
def detect_events_mv_mad(data: np.ndarray,
                         fs: int,
                         names: list[str],
                         mv_mads: np.ndarray,
                         mad_env: list[np.ndarray],
                         envs: tuple[list[np.ndarray],
                                     list[np.ndarray]],
                         env_percentile: int = 5,
                         mad_thrsh_f: float = 1.5,
                         env_thrsh_f: float = 2):
    mad_thresh = np.zeros(data.shape[0])
    # we'll write concurrently to the list and sort it afterwards
    rows = []
    for i in range(data.shape[0]):  # prange
        durations = []
        starts = []
        stops = []

        # the theshold for a peak/burst is a factor times a percentile of
        # the MAD signal envelope. The factor is given by the user, as well as
        # the percentile. The envelope percentile approach is chosen to detect
        # the noise level as the signal is very noisy and so is the moving MAD.
        mad_thresh[i] = (mad_thrsh_f * np.percentile(
                                mv_mads[i][mad_env[i]], env_percentile))

        # we have a peak/burst, if the mad is above the respective threshold
        above_thresh = (mv_mads[i] > mad_thresh[i]).astype(int)

        # find the indices where the signal crosses the threshold
        above_thresh = np.concatenate(([0], above_thresh, [0]))
        abs_diff = np.abs(np.diff(above_thresh))
        # and convert it to an array of tuples of the form (start, stop)
        above_thresh_idxs = np.where(abs_diff == 1)[0].reshape(-1, 2)
        for (start, stop) in above_thresh_idxs:
            durations.append((stop - start) / fs)
            starts.append(start)
            stops.append(stop)

        channel = np.repeat(names[i], len(durations))
        channel_events = pd.DataFrame(
                {"Channel": channel,
                 "StartIndex": starts,
                 "StopIndex": stops,
                 "Duration": durations}
                )
        rows.append(channel_events)

    return rows, mad_thresh


def compute_event_delays(rec: Recording):
    events_df = rec.events_df
    first_idx = events_df['StartIndex'].min()
    events_df['Delay'] = events_df['StartIndex'] - first_idx


def compute_event_tes(rec: Recording):
    events_df = rec.events_df
    sel_names = rec.get_sel_names()

    if 'Delay' not in events_df.columns:
        compute_event_delays(rec)

    fst_el_name = events_df[events_df['Delay'] == 0]['Channel'].iloc[0]
    fst_el_idx = sel_names.index(fst_el_name)
    fst_start_idx = events_df[events_df['Delay'] == 0]['StartIndex'].iloc[0]
    fst_stop_idx = events_df[events_df['Delay'] == 0]['StopIndex'].iloc[0]

    tes = []
    data = rec.get_data()
    first_signal = data[fst_el_idx][fst_start_idx:fst_stop_idx]
    for _, row in events_df.iterrows():
        el_name = row['Channel']
        el_idx = sel_names.index(el_name)
        start_idx = row['StartIndex']
        stop_idx = row['StopIndex']

        other_signal = data[el_idx][start_idx:stop_idx]
        tes.append(compute_transfer_entropy(first_signal, other_signal))

    events_df['TransferEntropy'] = tes


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


def extract_event_measures(data, electrode_idx, event_idxs, threshold):
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



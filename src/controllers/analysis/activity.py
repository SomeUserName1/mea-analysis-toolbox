import numpy as np
import pandas as pd
import scipy.signal as sg
from tqdm import tqdm
import pdb

from model.data import Recording, SharedArray
from constants import default_bins
from controllers.analysis.analyze import compute_entropies_jit
from controllers.analysis.spectral import bin_powers, compute_spectrograms


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
    if env_win is None:
        env_win = 0.1
    if env_percentile is None:
        env_percentile = 5
    if env_thrsh_f is None:
        env_thrsh_f = 2

    fs = rec.sampling_rate
    names = rec.get_sel_names()

    win = int(np.round(env_win * fs))
    compute_envelopes(rec, win)

    data = rec.get_data()
    n_peaks = np.zeros(data.shape[0])
    peaks_freq = np.zeros(data.shape[0])

    lower = np.zeros(data.shape[0])
    upper = np.zeros(data.shape[0])
    # we'll write concurrently to the list and sort it afterwards
    rows = []
    for i in tqdm(range(data.shape[0])):  # prange
        peaks = []
        peak_durations = []
        starts = []
        stops = []

        # Analogously, the thrshold for the signal amplitudes is based on a
        # percentile of the envelope of the signal itself with large window
        # (0.1s for example) multiplied by a facor. The factor is given by the
        # user, as well as the percentile.
        min_env = rec.envelopes[0][i]
        max_env = rec.envelopes[1][i]
        lower[i] = env_thrsh_f * np.percentile(data[i][min_env],
                                               (100 - env_percentile))
        upper[i] = env_thrsh_f * np.percentile(data[i][max_env],
                                               env_percentile)

        up_peaks, up_props = sg.find_peaks(data[i],
                                           height=upper[i],
                                           prominence=upper[i])

        up_widths = sg.peak_widths(data[i], up_peaks, rel_height=1)

        down_peaks, down_props = sg.find_peaks(-data[i],
                                               height=-lower[i],
                                               prominence=-lower[i])

        down_widths = sg.peak_widths(-data[i], down_peaks, 1)

        peaks = np.concatenate((up_peaks, down_peaks))

        if peaks.shape[0] == 0:
            continue

        order = np.argsort(peaks)
        peaks = peaks[order]

        peak_times = peaks / fs

        starts = np.concatenate((up_widths[2], down_widths[2]))
        starts = starts[order]

        stops = np.concatenate((up_widths[3], down_widths[3]))
        stops = stops[order]

        peak_durations = np.concatenate((up_widths[0], down_widths[0]))
        peak_durations = peak_durations[order] / fs

        peak_ampls = data[i][peaks] / np.abs(data[i][peaks]).max()

        n_peaks[i] = peaks.shape[0]
        peaks_freq[i] = n_peaks[i] / fs / 1000000

        channel = np.repeat(names[i], len(peaks))

        ipi = np.diff(peaks) / fs
        if peaks.shape[0] > 0:
            ipi = np.hstack((np.array([np.nan]), ipi))

        channel_peaks = pd.DataFrame(
                {"Channel": channel,
                 "PeakIndex": peaks,
                 "TimeStamp": peak_times,
                 "RelAmplitude": peak_ampls,
                 "StartIndex": starts,
                 "StopIndex": stops,
                 "Duration[s]": peak_durations,
                 "InterPeakInterval[s]": ipi}
                )
        rows.append(channel_peaks)

    rec.lower = lower
    rec.upper = upper

    # concatenate the list of data frames into one data frame,
    # sort it by channel and peak index and attach it to the recording object
    rec.peaks_df = pd.concat(rows)
    rec.peaks_df.sort_values(by=["Channel", "PeakIndex"], inplace=True)
    rec.channels_df['n_peaks'] = n_peaks
    rec.channels_df['peak_freq'] = peaks_freq


def detect_peaks_alt(rec: Recording,
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
    n_peaks = np.zeros(data.shape[0])
    peaks_freq = np.zeros(data.shape[0])

    lower = np.zeros(data.shape[0])
    upper = np.zeros(data.shape[0])
    mad_thresh = np.zeros(data.shape[0])
    # we'll write concurrently to the list and sort it afterwards
    rows = []
    for i in tqdm(range(data.shape[0])):  # prange
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
        min_env = rec.envelopes[0][i]
        max_env = rec.envelopes[1][i]
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

        peak_ampls = data[i][peaks] / np.abs(data[i][peaks]).max()
        channel = np.repeat(names[i], len(peaks))
        peak_times = peaks / fs

        ipi = np.diff(peaks) / fs
        if peaks.shape[0] > 0:
            ipi = np.hstack((np.array([np.nan]), ipi))

        channel_peaks = pd.DataFrame(
                {"Channel": channel,
                 "PeakIndex": peaks,
                 "TimeStamp": peak_times,
                 "RelAmplitude": peak_ampls,
                 "StartIndex": starts,
                 "StopIndex": stops,
                 "Duration[s]": peak_durations,
                 "InterPeakInterval[s]": ipi}
                )
        rows.append(channel_peaks)

    rec.lower = lower
    rec.upper = upper
    rec.mad_thresh = mad_thresh

    # concatenate the list of data frames into one data frame,
    # sort it by channel and peak index and attach it to the recording object
    rec.peaks_df = pd.concat(rows)
    rec.peaks_df.sort_values(by=["Channel", "PeakIndex"], inplace=True)
    rec.channels_df['n_peaks'] = n_peaks
    rec.channels_df['peak_freq'] = peaks_freq


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

    if rec.peaks_df is None:
        detect_peaks(rec)

    if rec.spectrograms is None:
        compute_spectrograms(rec)

    fs = rec.sampling_rate
    names = rec.get_sel_names()
    freq_bin_names = [f"{bins[0]}-{bins[1]}" for bins in default_bins]

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
    mad_thresh = np.zeros(data.shape[0])
    # we'll write concurrently to the list and sort it afterwards
    rows = []
    for i in tqdm(range(data.shape[0])):  # prange
        durations = []
        starts = []
        stops = []
        freqs = []
        n_peaks = []
        app_ens = []
        ipi = []

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

        # merge adjacent events when they are apart less than 100ms
        del_idxs = []
        idx_len = int(np.round(0.5 * fs))
        for j in range(1, above_thresh_idxs.shape[0]):
            if (above_thresh_idxs[j, 0]
                    - above_thresh_idxs[j - 1, 1] < idx_len):
                above_thresh_idxs[j, 0] = above_thresh_idxs[j - 1, 0]
                del_idxs.append(j - 1)

        above_thresh_idxs = np.delete(above_thresh_idxs, del_idxs, axis=0)

        # Drop all events that are shorter than 10ms and that don't have peaks
        del_idxs = []
        idx_len = int(np.round(0.128 * fs))
        for j in range(above_thresh_idxs.shape[0]):
            e_start = above_thresh_idxs[j, 0]
            e_stop = above_thresh_idxs[j, 1]
            if ((e_stop - e_start < idx_len)
                 or not any(data[i][e_start:e_stop] > rec.upper[i])
                 or not any(data[i][e_start:e_stop] < rec.lower[i])):
                del_idxs.append(j)
                
        above_thresh_idxs = np.delete(above_thresh_idxs, del_idxs, axis=0)

        for (start, stop) in above_thresh_idxs:
            durations.append((stop - start) / fs)
            starts.append(start)
            stops.append(stop)
            freqs.append(bin_powers(rec, i, (start, stop)))
            n_peaks.append(rec.peaks_df[(rec.peaks_df['Channel'] == names[i])
                                        & (rec.peaks_df['PeakIndex'] >= start)
                                        & (rec.peaks_df['PeakIndex'] < stop)
                                        ].shape[0]
                           )
            entropy = compute_entropies_jit(data[i][start:stop].reshape(1, -1))
            app_ens.append(entropy[0])
            ipi.append(rec.peaks_df[(rec.peaks_df['Channel'] == names[i])
                                    & (rec.peaks_df['PeakIndex'] >= start)
                                    & (rec.peaks_df['PeakIndex'] < stop)
                                    ]['InterPeakInterval[s]'].mean()
                       )
        iei = [start[i] - stop[i - 1] for i in range(1, len(starts))]
        if len(starts) > 0:
            iei.insert(0, np.nan)
        iei = np.array(iei) / fs

        channel = np.repeat(names[i], len(durations))
        channel_events = pd.DataFrame(
                {"Channel": channel,
                 "StartIndex": starts,
                 "StopIndex": stops,
                 "Duration [s]": durations,
                 "ApproximateEntropy": app_ens,
                 "#Peaks": n_peaks,
                 "MeanInterPeakInterval[s]": ipi,
                 "InterEventInterval[s]": iei,
                 } | dict(zip(freq_bin_names, np.array(freqs).T))
                )
        rows.append(channel_events)

    rec.event_mad_thresh = mad_thresh

    # concatenate the list of data frames into one data frame,
    # sort it by channel and peak index and attach it to the recording object
    rec.events_df = pd.concat(rows)


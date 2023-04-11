from abc import ABC


class Event(ABC):
    def __init__(self, electrode_idx, start_idx, end_idx, duration, spike_idxs, rms, max_amplitude, mean_isi, band_powers):
        self.electrode_idx = electrode_idx
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.duration = duration # in s
        self.spike_idxs = spike_idxs
        self.spike_count = len(spike_idxs[0])
        self.spike_rate = self.spike_count / duration
        self.rms = rms
        self.max_amplitude = max_amplitude
        self.mean_isi = mean_isi
        self.band_powers = band_powers
        self.delay = None
        self.te = None

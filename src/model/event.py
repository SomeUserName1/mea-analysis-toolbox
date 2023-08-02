"""
    Class respresenting a burst or seizure or more general a certain interval
    in the signal of an electrode along with some characteristics of it.
"""
class Event():
    def __init__(self, 
                 electrode_idx: int,
                 start_idx: int,
                 end_idx:int,
                 duration: float,
                 spike_idxs: list[int],
                 rms: float,
                 max_amplitude: float,
                 mean_isi: float,
                 band_powers: list[float]
                 ) -> None:
        """
            @param electrode_idx: index of the electrode in row major layout.
            @param start_idx: index of the beginning of the event in the data
                matrix.
            @param stop_idx: index of the end of the event in the data matrix.
            @param duration: in seconds.
            @param spike_idxs: indexes of spikes, i.e. where the signal is 
                above a certain threshold (one index per exceeding).
            @param rms: root mean squared of the signal in the time period.
                A meassure for the energy or power of the event.
            @param max_amplitude: The maximal amplitude of the event.
            @param mean_isi: Mean inter spike intervall
            @param band_powers: A list of values, one per band using the 
                standard bands (delta, theta, alpha, beta, gamma).
        """
        self.electrode_idx = electrode_idx
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.duration = duration # in s
        self.spike_idxs = spike_idxs
        self.spike_count = len(spike_idxs[0]) # number of spikes in event
        self.spike_rate = self.spike_count / duration
        self.rms = rms
        self.max_amplitude = max_amplitude
        self.mean_isi = mean_isi
        self.band_powers = band_powers
        self.delay = None
        self.te = None

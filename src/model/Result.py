import pint

class Result:
    def __init__(self,
                 date: str,
                 sampling_rate: int,
                 unit: pint.Unit,
                 selected_data: np.ndarray,
                 selected_names: list[str],
                 t_start: int,
                 t_stop: int,
                 ) -> None:
        """
        Data object used to hold the data matrix and metadata.

        @param date: the date when this recording was carried out.
        @param sampling rate: Sampling rate with which data was recorded.
        @param unit: unit of the data 
        @param data: the matrix holding the actual data.
            (num_channels, duration * sampling rate)
        """

        self.recording_date = date
        self.duration_mus = data.shape[1] / sampling_rate * 1000000
        self.sampling_rate = sampling_rate
        self.unit = unit
        self.data = selected_data
        self.names = selected_names
        self.t_start = 0
        self.t_start = self.duration_mus
        self.snrs = None # TODO self
        self.rms = None # impl
        self.derivative = None # TODO elephant
        self.mv_means = None # impl
        self.mv_stds = None # impl
        self.envelopes = None # impl
        self.psds = None # impl
        self.detrended_psds = None # TODO self
        self.fooof_group = None # impl
        self.spectrograms = None # impl
        self.entropies = None # TODO antropy
        self.peaks = None # TODO check and adapt
        self.events = None # TODO check and adapt
        self.isis = None # TODO check and adapt
        self.transfer_entopies = None # TODO check and adapt
        self.mutual_information = None # TODO minfo 
        self.xcorrs = None # TODO elephant, ca img script
        self.coherence = None # TODO elephant, scipy
        self.granger_caus = None # TODO elephant
        self.spectral_granger = None # TODO Elephant
        self.csds = None # TODO elephant
        self.psis = None # TODO finnpy
        self.pacs = None # TODO tensorpac
        # synchrony
        # phase synchrony
        # self.latencies c.f. intraop dataset repo

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
        self.snrs = None # ndarray (data.shape[0], 1)
        self.rms = None # ndarray (data.shape[0], 1)
        self.derivatives = None # ndarray (data.shape) 
        self.mv_means = None # ndarray (data.shape)
        self.mv_vars = None # ndarray (data.shape)
        self.mv_mads = None # ndarray (data.shape)
        self.envelopes = None # ndarray (data.shape)
        self.psds = None # tuple[ndarray (1,#freqs), ndarray(data.shape[0], #freqs) ]
        self.detrended_psds = None # ndarray(data.shape[0], #freqs)
        self.fooof_group = None # FOOOFGroup object
        self.spectrograms = None # ndarray (data.shape[0], time_res?)
        self.entropies = None # ndarray (data.shape[0], 1)
        self.peaks = None # list[ndarray (1, #peaks), dict (len=#peaks)] (len = data.shape[0])
        self.ipis = None # list[ndarray (1, #peaks -1)] 
        self.xcorrs = None # tuple[ ndarray (1, data.shape[1]), ndarray (data.shape[0], data.shape[0], data.shape[1])] 
        self.mutual_information = None # ndarray (data.shape[0], data.shape[0])  
        self.transfer_entopies = None # ndarray (data.shape[0], data.shape[0])
        self.coherence = None # TODO elephant, scipy
        self.granger_caus = None # TODO elephant
        self.spectral_granger = None # TODO Elephant
        self.csds = None # TODO elephant
        self.psis = None # TODO finnpy
        self.pacs = None # TODO tensorpac
        self.events = None # TODO check and adapt
        # synchrony
        # phase synchrony
        # self.latencies c.f. intraop dataset repo

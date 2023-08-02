import numpy as np
import pint

class Data:
    recording_date: str
    n_mea_electrodes: int
    duration_mus: int
    sampling_rate: int
    unit: pint.Unit
    data: numpy.ndarray 
    start_idx: int
    stop_idx: int
    electrode_names: list[str]

    def __init__(self,
                 date: str,
                 sampling_rate: int,
                 unit: pint.Unit,
                 data: np.ndarray,
                 start_idx: int,
                 stop_idx: int,
                 names: list[str]
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
       self.n_mea_electrodes = n_electrodes
       self.duration_mus = data.shape[1] / sampling_rate * 1000000
       self.sampling_rate = sampling_rate
       self.unit = unit
       self. data = data
       self.start_idx = start_idx
       self.stop_idx = stop_idx
       self.electrode_names = electrode_names
       self.selected_electrodes = []
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
        self.mutual_informations = None # ndarray (data.shape[0], data.shape[0])  
        self.transfer_entopies = None # ndarray (data.shape[0], data.shape[0])
        self.coherences = None # tuple[ndarray (#freqs), tuple[ndarray (1, #coherences), ndarray (1, #lags)]]
        self.granger_causalities = None # TODO list[list[dict]] (len(n_chanels), len(n_channels/2-1), caus_x_y, caus_y_x, instant_caus, total_dep)
        self.spectral_granger = None # freqs, as above
        self.csds = None # neo.AnalogSignal with estimated CSD
        self.events = None # TODO check and adapt
# self.psis = None # finnpy
# self.pacs = None # tensorpac
# synchrony
# phase synchrony
# self.latencies c.f. intraop dataset repo

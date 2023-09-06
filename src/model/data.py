import numpy as np
from multiprocessing.shared_memory import SharedMemory


class Recording:
    recording_date: str
    n_mea_electrodes: int
    duration_mus: int
    sampling_rate: int
    data: np.ndarray
    start_idx: int
    stop_idx: int
    electrode_names: list[str]

    def __init__(self,
                 fname: str,
                 date: str,
                 n_electrodes: int,
                 sampling_rate: int,
                 data: np.ndarray,
                 start_idx: int,
                 stop_idx: int,
                 names: np.ndarray,
                 ground_els: np.ndarray,
                 ground_el_names: np.ndarray
                 ) -> None:
        """
        Data object used to hold the data matrix and metadata.

        @param date: the date when this recording was carried out.
        @param sampling rate: Sampling rate with which data was recorded.
        @param data: the matrix holding the actual data.
            (num_channels, duration * sampling rate)
        """
        self.fname = fname
        self.recording_date = date
        self.n_mea_electrodes = n_electrodes
        self.duration_mus = data.shape[1] / sampling_rate * 1000000
        self.start_idx = start_idx
        self.stop_idx = stop_idx
        self.electrode_names = names
        self.ground_els = ground_els
        self.ground_el_names = ground_el_names
        self.selected_electrodes = []
        self.sampling_rate = sampling_rate

        # create shared memory
        self.data = SharedArray(data)

        # Maybe used for burst detection and burst & peak characterization
        self.derivatives = None # ndarray (data.shape)
        self.mv_means = None # ndarray (data.shape)
        self.mv_vars = None # ndarray (data.shape)
        self.mv_mads = None # ndarray (data.shape)
        self.envelopes = None # ndarray (data.shape)
        ###### Spectral --- Store output of fooof wrt. psd. May use spectrogram as fooof group
        self.psds = None # tuple[ndarray (1,#freqs), ndarray(data.shape[0], #freqs) ]
        self.detrended_psds = None # ndarray(data.shape[0], #freqs)
        self.fooof_group = None # FOOOFGroup object
        self.spectrograms = None # freqs, ts, ndarray (data.shape[0], freqs, time_res?)
        ####### Not sure how to put that into df. maybe max for xcorr, for coherence peaks and so on. check papers
        self.xcorrs = None # tuple[ ndarray (1, data.shape[1]), ndarray (data.shape[0], data.shape[0], data.shape[1])]
        self.mutual_informations = None # ndarray (data.shape[0], data.shape[0])
        self.transfer_entopies = None # ndarray (data.shape[0], data.shape[0])
        self.coherences = None # tuple[ndarray (#freqs), tuple[ndarray (#coherences), ndarray (#lags)]]
        self.granger_causalities = None # list[list[dict]] (len(n_chanels), len((n_channels-1)/2), caus_x_y, caus_y_x, instant_caus, total_dep)
        self.spectral_granger = None # freqs, as above
        self.csds = None # neo.AnalogSignal with estimated CSD
        self.channels_df = None # Cols: SNR, RMS, Apprx_Entropy, n_peaks, firing rate
        self.peaks_df = None
        self.bursts_df = None
        self.seizures_df = None
        self.network_df = None
# self.psis = None # finnpy
# self.pacs = None # tensorpac
# synchrony
# phase synchrony
# self.latencies c.f. intraop dataset repo

    def get_sel_names(self):
        return self.electrode_names[self.selected_electrodes]

    def get_time_s(self):
        t_start = self.start_idx / self.sampling_rate
        t_stop = self.stop_idx / self.sampling_rate

        return t_start, t_stop

    def get_data(self):
        return self.data.read()

    def free(self):
        self.data.free()


class SharedArray:
    '''
    Wraps a numpy array so that it can be shared quickly among processes,
    avoiding unnecessary copying and (de)serializing.
    '''

    def __init__(self, array: np.ndarray, dtype=None):
        '''
        Creates the shared memory and copies the array therein

        :param array: the array to be shared
        :type array: np.ndarray
        '''
        # create the shared memory location of the same size of the array
        self._shared = SharedMemory(create=True, size=array.nbytes)
        self._name = self._shared.name

        # save data type and shape, necessary to read the data correctly
        self._dtype = array.dtype if dtype is None else dtype
        self._shape = array.shape

        # create a new numpy array that uses the shared memory we created.
        # at first, it is filled with zeros
        res = np.ndarray(self._shape, self._dtype, buffer=self._shared.buf)
        # copy data from the array to the shared memory. numpy will
        # take care of copying everything in the correct format
        res[:] = array[:]

    def read(self):
        '''
        Reads the array from the shared memory without unnecessary copying.
        '''
        # open the shared memory region and simply create an array of the
        # correct shape and type
        return np.ndarray(self._shape, self._dtype, buffer=self._shared.buf)

    def close(self):
        '''
        Closes the shared memory region.
        '''
        self._shared.close()

    def free(self):
        '''
        Closes and unlinks the shared memory region.
        '''
        self._shared.close()
        self._shared.unlink()

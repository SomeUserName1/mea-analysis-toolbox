import numpy as np

class Data:
    def __init__(self,
                 date: str,
                 sampling_rate: int,
                 units: np.ndarray,
                 data: np.ndarray
                 ) -> None:
        """
        Data object used to hold the data matrix and metadata.

        @param date: the date when this recording was carried out.
        @param sampling rate: Sampling rate with which data was recorded.
        @param unit: unit of the data per channel. (1, num_channels)
        @param data: the matrix holding the actual data. 
            (num_channels, duration * sampling rate)
        """
        self.recording_date = date
        self.num_electrodes = data.shape[0]
        self.duration_mus = data.shape[1] / sampling_rate * 1000000
        self.sampling_rate = sampling_rate
        self.units = units
        self.data = data
        self.selection_applied = False
        self.selected_rows = []
        self.events = None

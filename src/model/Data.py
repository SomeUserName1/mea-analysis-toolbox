import numpy as np
import pint

class Data:
    def __init__(self,
                 date: str,
                 sampling_rate: int,
                 unit: pint.Unit,
                 data: np.ndarray,
                 ground_els: list[int] = None
                 ) -> None:
        """
        Data object used to hold the data matrix and metadata.

        @param date: the date when this recording was carried out.
        @param sampling rate: Sampling rate with which data was recorded.
        @param unit: unit of the data 
        @param data: the matrix holding the actual data.
            (num_channels, duration * sampling rate)
        """
        side_len = int(np.sqrt(data.shape[0]))
        names = [f"R{i}C{j}" for i in range(1, side_len + 1) for j in range(1, side_len + 1)]

        self.recording_date = date
        self.num_electrodes = data.shape[0]
        self.duration_mus = data.shape[1] / sampling_rate * 1000000
        self.sampling_rate = sampling_rate
        self.unit = unit
        self.data = data
        self.electrode_names = names
        self.selected_electrodes = []
        self.ground_electrodes = ground_els
        self.start_idx = 0
        self.stop_idx = self.duration_mus
        self.events = None


    def get_selected(self) -> np.ndarray:
        """
        Returns the data matrix with electrode selection and time window
        applied.
        """
        return self.data[self.selected_rows, self.start_idx:self.stop_idx]


    def set_time_window(self, start_mus: int, stop_mus: int) -> None:
        """
        Sets the time window of the data to be evaluated & displayed.
        """
        start_idx = int(np.round(self.sampling_rate * start_mus / 1000000))
        stop_idx = int(np.round(self.sampling_rate * stop_mus / 1000000))

        self.start_idx = start_idx
        self.stop_idx = stop_idx

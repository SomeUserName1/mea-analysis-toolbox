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
        names = [f"R{i}C{j}" for i in range(1, side_len + 1) \
                    for j in range(1, side_len + 1)]

        self.recording_date = date
        self.num_electrodes = data.shape[0]
        self.duration_mus = data.shape[1] / sampling_rate * 1000000
        self.sampling_rate = sampling_rate
        self.unit = unit
        self.data = data
        self.electrode_names = names
        self.ground_electrodes = ground_els
        self.selected_electrodes = []
        self.start_idx = 0
        self.stop_idx = self.duration_mus
        self.selected_data = None


    def get_selected_names(self) -> np.ndarray:
        """
        Returns the data matrix with electrode selection and time window
        applied.
        """
        return self.names[self.selected_rows]




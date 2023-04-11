"""
Data object used to hold the data matrix and metadata
"""
from abc import ABC


class Data(ABC):
    def __init__(self, date, num_electrodes, sampling_rate, data):
        self.recording_date = date
        self.num_electrodes = num_electrodes
        self.duration_mus = data.shape[1] / sampling_rate * 1000000
        self.sampling_rate = sampling_rate
        self.data = data
        self.selection_applied = False
        self.selected_rows = []
        self.events = None

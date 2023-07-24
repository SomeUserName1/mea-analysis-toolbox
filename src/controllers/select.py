import math
from multiprocessing import Process
import re
from typing import Optional

import numpy as np
from wand.image import Image

from constants import img_size
from model.Data import Data


def update_selected_data(data: Data) -> None:
    """
    Creates a new NumPy array with the rows and their elements corresponding 
        to the selected electrodes and the selected time window only.
    """
    data.selected_data = data.data[data.selected_rows, data.start_idx:data.stop_idx]


def set_time_window(data: Data, start_mus: int, stop_mus: int) -> None:
    """
    Sets the time window of the data to be evaluated & displayed.
    """
    start_idx = int(np.round(data.sampling_rate * start_mus / 1000000))
    stop_idx = int(np.round(data.sampling_rate * stop_mus / 1000000))

    data.start_idx = start_idx
    data.stop_idx = stop_idx

def update_time_window(data: Data, t_start: str, t_end: str) -> None:
    """
    Converts the string time gotten from the ui in format s:ms:mus to micro
    seconds.
    Used to for the time window selection.

    @param t_start: start time as string in the above format.
    @param t_end: end time as string in the above format.
    @param duration: total duration of the recording in micro seconds

    @return the start and stop points in micro seconds as an integer.
    """
    if t_start is not None and t_start != "":
        start_cut = str_to_mus(t_start)

        if start_cut < 0 or start_cut > data.duration_mus:
            raise RuntimeError(("The start point of the selected time window"
                                "needs to be larger than 0 and be within the "
                                "recording duration"))
    else:
        start_cut = 0

    if t_end is not None and t_end != "":
        end_cut = str_to_mus(t_end)

        if t_start is not None and start_cut > end_cut:
            raise RuntimeError(("The end point of the selected time window "
                                "must be smaller than the start point!"))
    else:
        end_cut = data.duration_mus

    data.set_time_window(start_cut, end_cut)


def update_electrode_selection(data: Data,
                               selected_electrodes: Optional[list[int]],
                               clicked_electrode: Optional[int]
                               ) -> None:
    """
    Updates the list of selected electrode as a call back to a click in the
    electrode grid window in the selection view.

    @param data: the data holder object
    @param selected_electrodes: Electrodes which were selected by box or
        lasso selection in the selection view.
    @param clicked_electrodes: Electrodes selected by clicking a single one
        in the selection view.
    """
    grid_size = int(np.sqrt(data.num_electrodes))
    electrodes = []

    if clicked_electrode is not None:
        electrodes.append(clicked_electrode['points'][0])
    if selected_electrodes is not None:
        electrodes.extend(selected_electrodes['points'])
    for point in electrodes:
        match = re.match(r"R(?P<row>\d+)C(?P<col>\d+)", point['text'])
        # As we label the electrodes for non computer scientists (aka starting
        # from 1), we need to subtract one to get the correct index.
        idx = (int(match['row']) - 1) * grid_size + int(match['col']) - 1

        # Check if electrode is in selected list.
        # If it is already remove it.
        if idx in data.selected_electrodes:
            data.selected_electrodes.remove(idx)
        # If the newly selected electrode is not a ground electrode add it.
        elif idx not in data.ground_electrodes:
            # if it's not qa ground electrode add it
            data.selected_electrodes.append(idx)

    data.selected_electrodes.sort()


def max_duration(data: Data) -> tuple[int, int, int]:
    """
    Converts the duration of the recording in micro seconds to a tuple in
    format s:ms:mus.

    @param duration: duration of the recording in micro seconds

    @return the duration as a string in the above format.
    """
    time_mus = data.duration_mus % 1000
    time_ms = math.floor(data.duration_mus / 1000) % 1000
    time_s = math.floor(data.duration_mus / 1000000)

    return time_s, time_ms, time_mus


def str_to_mus(smsmus: str) -> int:
    """
    Converts seconds, milli seconds and micro seconds to all micro seconds.

    @param s: seconds specified in input
    @param ms: seconds specified in input
    @param mus: seconds specified in input

    @return added seconds, milliseconds and microseconds all in micro seconds
    """
    if len(t_start.split(":")) != 3:
        raise RuntimeError(("The start point of the selected time window"
                            "needs to be in the format s:ms:mus"))
    s, ms, mus = smsmus.split(":")
    return int(s) * 1000000 + int(ms) * 1000 + int(mus)


def convert_to_jpeg(image_path: str) -> str:
    """
    Converts an image from arbitrary format to a jpeg image URL for displaying
    in the selection view.

    @param image_path: Path of the image to convert.

    @return jpeg image URL
    """
    # FIXME Add RuntimeError if image_path is not a valid path
    img = Image(filename=image_path)
    img.format = 'jpeg'
    img.sample(img_size, img_size)
    img_url = img.data_url()

    return img_url

"""
TODO
"""
import math
from typing import Optional

import numpy as np
import pandas as pd
from wand.image import Image

from constants import img_size
from model.data import Recording, SharedArray
from views.grid_plot_utils import el_idx_plot_to_data


def apply_selection(rec: Recording):
    """
    Creates a new NumPy array with the rows and their elements corresponding
        to the selected electrodes and the selected time window only.

    :param rec: the recording object
    :type rec: Recording
    """
    prev_data = rec.data
    data = rec.get_data()
    rec.data = SharedArray(data[rec.selected_electrodes,
                                rec.start_idx:rec.stop_idx])
    prev_data.free()
    rec.channels_df = pd.DataFrame(rec.get_sel_names(), columns=['Channel'],
                                   dtype="string")


def update_time_window(rec: Recording, t_start: str, t_end: str):
    """
    Converts the string time gotten from the ui in format s:ms:mus to micro
    seconds.
    Used to for the time window selection.

    :param t_start: start time as string in the above format.
    :type t_start: str

    :param t_end: end time as string in the above format.
    :type t_end: str
    """
    if t_start is not None and t_start != "":
        start_mus = str_to_mus(t_start)

        if start_mus < 0 or start_mus > rec.duration_mus:
            raise RuntimeError(("The start point of the selected time window"
                                "needs to be larger than 0 and be within the "
                                "recording duration"))
    else:
        start_mus = 0

    if t_end is not None and t_end != "":
        stop_mus = str_to_mus(t_end)

        if t_start is not None and start_mus > stop_mus:
            raise RuntimeError(("The end point of the selected time window "
                                "must be smaller than the start point!"))
    else:
        stop_mus = rec.duration_mus

    start_idx = int(np.round(rec.sampling_rate * start_mus / 1000000))
    stop_idx = int(np.round(rec.sampling_rate * stop_mus / 1000000))

    rec.start_idx = start_idx
    rec.stop_idx = stop_idx


def update_electrode_selection(rec: Recording,
                               selected_electrodes: Optional[list[int]],
                               clicked_electrode: Optional[int]
                               ) -> None:
    """
    Updates the list of selected electrode as a call back to a click in the
    electrode grid window in the selection view.

    :param data: the recording object
    :type rec: Recording

    :param selected_electrodes: Electrodes which were selected by box or
        lasso selection in the selection view.
    :type selected_electrodes: list[int]

    :param clicked_electrodes: Electrodes selected by clicking a single one
        in the selection view.
    :type clicked_electrodes: int
    """
    grid_size = int(np.sqrt(rec.n_mea_electrodes))
    electrodes = []
    if clicked_electrode is not None:
        electrodes.append(clicked_electrode['points'][0])
    if selected_electrodes is not None:
        electrodes.extend(selected_electrodes['points'])
    for point in electrodes:
        # As we label the electrodes for non computer scientists (aka starting
        # from 1), we need to subtract one to get the correct index.
        coords = [int(s)-1 for s in point['text'].split() if s.isdigit()]
        idx = coords[0] * grid_size + coords[1]
        idx = el_idx_plot_to_data(rec, idx)

        # Check if electrode is in selected list.
        # If it is already remove it.
        if idx in rec.selected_electrodes:
            rec.selected_electrodes.remove(idx)
        elif idx == -1:  # is a ground electrode
            continue
        # Else add it.
        else:
            rec.selected_electrodes.append(idx)

    rec.selected_electrodes.sort()


def max_duration(rec: Recording) -> tuple[int, int, int]:
    """
    Converts the duration of the recording in micro seconds to a tuple in
    format s:ms:mus.

    :param duration: duration of the recording in micro seconds
    :type duration: int

    :return the duration as a string in the above format.
    :type return: tuple[int, int, int]
    """
    time_mus = rec.duration_mus % 1000
    time_ms = math.floor(rec.duration_mus / 1000) % 1000
    time_s = math.floor(rec.duration_mus / 1000000)

    return time_s, time_ms, time_mus


def str_to_mus(smsmus: str) -> int:
    """
    Converts seconds, milli seconds and micro seconds to all micro seconds.

    :param s: seconds specified in input
    :type s: int

    :param ms: seconds specified in input
    :type ms: int

    :param mus: seconds specified in input
    :type mus: int

    :return added seconds, milliseconds and microseconds all in micro seconds
    :type return: int
    """
    if len(smsmus.split(":")) != 3:
        raise RuntimeError(("The start point of the selected time window"
                            "needs to be in the format s:ms:mus"))
    s, ms, mus = smsmus.split(":")
    return int(s) * 1000000 + int(ms) * 1000 + int(mus)


def convert_to_jpeg(image_path: str) -> str:
    """
    Converts an image from arbitrary format to a jpeg image URL for displaying
    in the selection view.

    :param image_path: Path of the image to convert.
    :type image_path: str

    :return jpeg image URL
    :type return: str
    """
    # FIXME Add RuntimeError if image_path is not a valid path
    img = Image(filename=image_path)
    img.format = 'jpeg'
    img.sample(img_size, img_size)
    img_url = img.data_url()

    return img_url

import math
from multiprocessing import Process

import numpy as np
from wand.image import Image

from views.electrode_grid import grid_size, draw_electrode_grid, align_grid_to_rows
from views.time_series_plot import plot_in_grid


img_size = int(0.7 * grid_size)


def convert_time(t_start: str, t_end: str, duration: int) -> tuple[int, int]:
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
        s_start, ms_start, mus_start = t_start.split(":")
        start_cut = convert_to_mus(s_start, ms_start, mus_start)

        if start_cut < 0 or start_cut > data.duration_mus:
            raise RuntimeError(("The start point of the selected time window"
                                "needs to be larger than 0 and be within the "
                                "recording duration"))
    else:
        start_cut = None

    if t_end is not None and t_end != "":
        s_end, ms_end, mus_end = t_end.split(":")
        end_cut = convert_to_mus(s_end, ms_end, mus_end)

        if t_start is not None and start_cut > end_cut:
            raise RuntimeError(("The end point of the selected time window "
                                "must be smaller than the start point!"))
    else:
        end_cut = None

    return start_cut, end_cut


def apply_selection(data: Data, t_start: int, t_stop: int) -> None:
    """
    Discard data besides the selected channels and time window.
    Data could also be kept, this is mainly to speed up the subsequent
    computations.

    @param data: object holding raw data and meta informations.
    @param t_start: beginning of the selected time window in micro seconds.
    @param t_stop: end of the selected time window in micro seconds.

    @return the next button to indicate completion and make the preprocessing
        view available.
    """
    if t_start is None:
        t_start = 0
    if t_stop is None:
        t_stop = data.duration_mus

    rows = sorted(data.selected_rows)

    if len(rows) == 0:
        return

    data.data = data.data[rows]

    t_start_idx = int(np.round(data.sampling_rate * t_start / 1000000))
    t_stop_idx = int(np.round(data.sampling_rate * t_stop / 1000000))

    data.data = data.data[:, t_start_idx:t_stop_idx]

    data.duration_mus = t_stop - t_start
    data.selection_applied = True


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
    name_row_map = mcs_256mea_get_name_row_map()
    electrodes = []

    if clicked_electrode is not None:
        electrodes.append(clicked_electrode['points'][0])
    if selected_electrodes is not None:
        electrodes.extend(selected_electrodes['points'])

    for point in electrodes:
        # Check if electrode is in selected list.
        row = name_row_map[point['text']]

        if row in data.selected_rows:
            data.selected_rows.remove(row)
        else:
            # Otherwise add it
            data.selected_rows.append(row)


def plot_selected_rows(data: Data, start: int, end: int) -> None:
    """
    Plots the data for the selected time window and electrodes as specified in
    the select view.

    @params data: Data holder object
    @params start: start of the selected time window in micro seconds.
    @params end: end of the selected time window in micro seconds.
    """
    el_idx_sort = sorted(data.selected_electrodes)
    names_selected_sorted = mcs_256mea_get_names()[el_idx_sort]

    if len(data.selected_rows) == 0:
        return

    if start is None:
        start = 0
    if end is None:
        end = data.duration_mus

    t_start_idx = int(np.round(data.sampling_rate * start / 1000000))
    t_end_idx = int(np.round(data.sampling_rate * end / 1000000))
    if data.selection_applied:
        signals = data.data[:, t_start_idx:t_end_idx]
    else:
        signals = data.data[data.selected_rows, t_start_idx:t_end_idx]


    proc = Process(target=plot_in_grid, args=('time_series', signals, data.selected_rows, \
            names_selected_sorted, data.sampling_rate, start, end))
    proc.start()
    proc.join()


def duration_to_str(data: Data) -> str:
    time_mus = data.duration_mus % 1000
    time_ms = math.floor(data.duration_mus / 1000) % 1000
    time_s = math.floor(data.duration_mus / 1000000)

    return time_s, time_ms, time_mus


def convert_to_mus(s: int, ms: int, mus: int):
    """
    Converts seconds, milli seconds and micro seconds to all micro seconds.

    @param s: seconds specified in input
    @param ms: seconds specified in input
    @param mus: seconds specified in input

    @return added sseconds, milliseconds and microseconds all in micro seconds
    """
    return int(s) * 1000000 + int(ms) * 1000 + int(mus)


def convert_to_jpeg(image_path: str) -> str:
    """
    Converts an image from arbitrary format to a jpeg image URL for displaying
    in the selection view.

    @param image_path: Path of the image to convert.

    @return jpeg image URL
    """
    img = Image(filename=image_path)
    img.format = 'jpeg'
    img.sample(img_size, img_size)
    img_url = img.data_url()

    return img_url

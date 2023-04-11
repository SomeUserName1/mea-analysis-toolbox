import math
from multiprocessing import Process

import numpy as np
from wand.image import Image

from ui.select import next_button, no_data
from views.electrode_grid import grid_size, draw_electrode_grid, align_grid_to_rows
from views.time_series_plot import plot_in_grid
from model.io.import_mcs import get_mcs_256mea_name_row_dict, \
                                get_mcs_256mea_row_name_dict, \
                                get_selected_electrode_names


img_size = int(0.7 * grid_size)


def convert_time(t_start, t_end, data):
    if t_start is not None and t_start != "":
        s_start, ms_start, mus_start = t_start.split(":")
        start_cut = convert_to_mus(s_start, ms_start, mus_start)
       
        if start_cut < 0 or start_cut > data.duration_mus:
            raise RuntimeError("The start point of the selected time window"
                   + " needs to be larger than 0 and be within the recording duration")
    else:
        start_cut = None

    if t_end is not None and t_end != "":
        s_end, ms_end, mus_end = t_end.split(":")
        end_cut = convert_to_mus(s_end, ms_end, mus_end)

        if t_start is not None and start_cut > end_cut:
            raise RuntimeError("The end point of the selected time window "
                    + "must be smaller than the start point!")
    else:
        end_cut = None

    return start_cut, end_cut


def apply_selection(data, t_start, t_stop):
    if t_start is None:
        t_start = 0
    if t_stop is None:
        t_stop = data.duration_mus

    rows = sorted(data.selected_rows)

    if len(rows) == 0:
        return no_data

    data_mat = data.data[rows]

    t_start_idx = int(np.round(data.sampling_rate * t_start / 1000000))
    t_stop_idx = int(np.round(data.sampling_rate * t_stop / 1000000))

    data_mat = data_mat[:, t_start_idx:t_stop_idx]

    data.data = data_mat
    data.duration_mus = t_stop - t_start
    data.selection_applied = True

    return next_button


def update_electrode_selection(data, selected_electrodes, clicked_electrode):
    name_row_map = get_mcs_256mea_name_row_dict()
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


def update_grid(data):
    # replot grid with lists
    x_sel, y_sel, x_un, y_un, names_sel, names_un, _ = \
            align_grid_to_rows(data)

    grid = draw_electrode_grid(16, x_sel, y_sel, names_sel, x_un, y_un, \
                               names_un)

    # return el grid, single selected and all selected
    return grid

def plot_selected_rows(data, start, end):
    names_selected_sorted = get_selected_electrode_names(data)

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


def max_time(data):
    time_mus = data.duration_mus % 1000
    time_ms = math.floor(data.duration_mus / 1000) % 1000
    time_s = math.floor(data.duration_mus / 1000000)

    return time_s, time_ms, time_mus


def convert_to_mus(s, ms, mus):
    return int(s) * 1000000 + int(ms) * 1000 + int(mus)


def convert_to_jpeg(image_path):
    img = Image(filename=image_path)
    img.format = 'jpeg'
    img.sample(img_size, img_size)
    img_url = img.data_url()

    return img_url

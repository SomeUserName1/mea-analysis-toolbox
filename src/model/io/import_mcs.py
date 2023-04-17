"""
Importer for multi channel system MEAs, assuming all data is in a single \
        analog stream in the same recording.
"""
import datetime
import os.path

import McsPy
import McsPy.McsData
import numpy as np
from tabulate import tabulate
import scipy.signal as sg

from model.Data import Data
from controllers.preproc import downsample


def import_mcs(path, que):
    """
    Import data recorded with a MultiChannel Systems MEA into A Data object, \
            see model/Data.py.

        @param path: the path to the file containing the data in McS h5 format.

        @return a Data object containing the data in-memory and metadata or \
                None and an error message
    """
    if path is None or not os.path.exists(path):
        return None, "File does not exist or invalid path!"

    McsPy.McsData.VERBOSE = False
    try:
        file_contents = McsPy.McsData.RawData(path)
        date = file_contents.date
        stream = file_contents.recordings[0].analog_streams[0]
        num_channels = stream.channel_data.shape[0]
        channel_row_map, sampling_rate = \
                build_channel_map(file_contents)
        data = np.array(stream.channel_data)

        for i in [c.channel_id for c in stream.channel_infos.values()]:
            data[channel_row_map[i]] -= \
                    stream.channel_infos[i].get_field('ADZero')

        data = sort_data_row_major(data)

        info = print_info(path, file_contents)
        data = Data(date, num_channels, sampling_rate, data)
        del file_contents

    except IOError as err:
        info = "Failed to import specified file! Please specify a valid" \
                + " multi channel systems H5 formatted file.\n" \
                + "Error: " + str(err)
        data = None

    que.put((data, info))


def build_channel_map(raw_data):
    """
    Constructs two maps mapping from channel id to row in the data matrix and \
            vice versa. Also verifies the sampling rate across channels to be \
            the same.

        @param raw_data: the object returned by calling McsPy.McData.RawData

        @return maps from channel id to row in the data matrix and vice versa \
                and the sampling rate.
    """
    id_row_map = {}

    stream = raw_data.recordings[0].analog_streams[0]
    sampling_rate = stream.channel_infos[2].sampling_frequency.magnitude

    for i in [c.channel_id for c in
              stream.channel_infos.values()]:
        row = stream.channel_infos[
            i].row_index
        assert (stream.channel_infos[
                    i].sampling_frequency.magnitude
                == sampling_rate)
        id_row_map[i] = row

    return id_row_map, sampling_rate


def print_header_info(h5filename, data):
    """
    Prints infos that are contained in the header of the McS h5 file, like \
            the MEA name, the version, ...

            @param h5filename: Name of the file containing the data.
            @param data: : the object returned by calling McsPy.McData.RawData

        @return the information formatted as a table
    """
    header_info = "\nFile path:" + h5filename + "\n\n"
    t_row = []
    date = datetime.datetime(1, 1, 1) + datetime.timedelta(
        microseconds=int(data.date_in_clr_ticks) / 10)
    t_row.append(str(date.strftime("%Y-%m-%d %H:%M:%S")))
    t_row.append(data.program_name)
    t_row.append(data.program_version)
    t_row.append(data.comment)
    t_row.append(data.mea_name)
    t_row.append(data.mea_layout)
    real_row = [t_row]
    table_header = ["Date", "Program", "Version", "Comment", "MEA System Name",
                    "MEA Layout"]

    return header_info + tabulate(real_row, headers=table_header)


def print_info(h5filename, data):
    """
    Prints infos about the McS h5 file and the available stream(s)

        @param h5filename: Name of the file containing the data.
        @param data: : the object returned by calling McsPy.McData.RawData

        @return the information formatted as a table
    """
    info_string = print_header_info(h5filename, data) + "\n\n"
    recording = data.recordings[0]

    if recording is None:
        return ""

    all_rows = []
    table_header = ["Type", "Stream", "# ch"]

    streams = vars(recording).items()
    for key, value in streams:
        if value is None or key not in ["_Recording__analog_streams",
                                  "_Recording__frame_streams",
                                  "_Recording__event_streams",
                                  "_Recording__segment_streams",
                                  "_Recording__timestamp_streams"]:
            continue

        for _, stream in value.items():
            row = [stream.stream_type, stream.label]

            try:
                row.append(len(stream.channel_infos))
            except AttributeError:
                row.append("")

            all_rows.append(row)

    return info_string + tabulate(all_rows, headers=table_header)

# FIXME sth is off here.
def extract_baseline_std(baseline, que):
    """
    Extracts the standard deviation per channel from a file without creating a \
            Data object.

        @param baseline: path to the McS h5 file with the baseline data.
        @param selected_rows: the rows of the selected electrodes of the \
                stimulus data (see detect_peaks/detect_events in controllers/ \
                analyze.py)
        @param moving_avg: boolean indicating if the std of the moving \
                average shall be extracted.

        @return the std of the channel or of the moving average of the channel
    """
    file_contents = McsPy.McsData.RawData(baseline)
    stream = file_contents.recordings[0].analog_streams[0]
    num_electrodes = stream.channel_data.shape[0]
    sampling_rate = stream.channel_infos[2].sampling_frequency.magnitude
    data = stream.channel_data
    data = np.array(data)
    # downsample signal to speed things up
    q = int(np.round(sampling_rate / 1000))
    q_it = 0
    for i in range(12):
        if q % (12 - i) == 0:
            q_it = 12 - i
            break

    if q_it == 0:
        q_it = 10

    i = 0
    while q > 13:
        q = int(np.round(q / q_it))
        data = sg.decimate(data, q_it)
        i += 1

    sampling_rate = sampling_rate / q_it ** i

    q = int(np.floor(q))
    if q != 0:
        data = sg.decimate(data, q)
        sampling_rate = sampling_rate / q
    stds = []
    mv_std_stds = []
    mv_mad_stds = []
    window = int(np.round(data.shape[1] / 20))

    for i in range(num_electrodes):
        abs_dev = np.abs(data[i] - np.mean(data[i]))
        mv_mad = np.convolve(abs_dev, np.ones(window), 'same') / window

        mv_std = np.convolve(np.square(abs_dev), np.ones(window), 'same') / window
        mv_std = np.sqrt(mv_std)
        
        mv_std_stds.append(np.std(mv_std))
        mv_mad_stds.append(np.std(mv_mad))
        stds.append(np.std(data[i]))

    del file_contents

    que.put((stds, mv_std_stds, mv_mad_stds))


def sort_data_row_major(data: np.ndarray) -> np.ndarray:
    order = get_mcs_256mea_row_order()

    data = data[order]
    
    for i in [0, 15, 239, 255]:
        np.insert(ordered_data, i, np.nan, axis=0)
    
    return data


def get_mcs_256mea_names():
    """
    Constructs the list of channel names for the 256 electrode MEA system as \
            specified in the manual.
    """
    names = []

    for number in range(1, 17):
        for letter in ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L",
                       "M", "N", "O", "P", "R"]:
            if number in [1, 16] and letter in ['A', 'R']:
                continue

            names.append(letter + str(number))

    return names

def get_mcs_256mea_row_order():
    """
    Order the channels as on the MEA. As the channel that corresponds to row \
            0 is not in the top left corner but on an arbeitrary position, \
            this is necessary to display the data for a certain channel \
            name correctly.
    """
    row_order = None

    with open("assets/mcs_256mea_mapping.txt", "r") as ids_file:
        row_order = np.array([int(v) for v in ids_file.read().split(",") \
                if v.strip() != ''])

    return row_order


def get_mcs_256mea_row_name_dict():
    """
    returns a map that maps from channel if to the name.
    """
    names = get_mcs_256mea_names()
    channel_order = get_mcs_256mea_row_order()

    return dict(zip(channel_order, names))


def get_mcs_256mea_name_row_dict():
    """
    Returns a map from channel names to channel ids
    """
    names = get_mcs_256mea_names()
    channel_order = get_mcs_256mea_row_order()

    return dict(zip(names, channel_order))


def get_selected_electrode_names(data):
    """
    Returns the names of the selected electrodes

        @param data:  data object, see model/Data.py

        @return list of names of the selected electrodes
    """
    names_rows = get_mcs_256mea_row_name_dict()
    names = []
    if data.selected_rows is not None:
        names = [names_rows[row_id] for row_id in data.selected_rows]
    return names

"""
:author: Fabian Klopfer <fabian.klofper@ieee.org>
:date:   21.04.2023

Importer for multi channel systems 256MEAs chip, assuming all data is in a single \
        analog stream in the same recording.
"""
import datetime
from multiprocessing import Queue
import os.path

import McsPy
import McsPy.McsData
import numpy as np
from tabulate import tabulate
import scipy.signal as sg

from model.Data import Data
from controllers.preproc import downsample


def mcs_256mea_import(path: str, que: Queue) -> None:
    """
    Import data recorded with a MultiChannel Systems MEA into A Data object, \
            see model/Data.py.

        @param path: the path to the file containing the data in McS h5 format.
        @param que: A python Queue, to return the loaded data asynchronously.

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
        sampling_rate = stream.channel_infos[2].sampling_frequency.magnitude
        data = np.array(stream.channel_data)

        units = np.empty(num_channels)
        channel_row_map = {}
        row_order = None

        for i in [c.channel_id for c in stream.channel_infos.values()]:
            row = stream.channel_infos[i].row_index
            id_row_map[i] = row

        # correct signal values, see MCS implementation:
        # https://mcspydatatools.readthedocs.io/en/latest/api.html#McsPy.McsData.AnalogStream.get_channel_in_range
        for i in [c.channel_id for c in stream.channel_infos.values()]:
            adc_step = self.channel_infos[i].adc_step.magnitude
            ad_zero = stream.channel_infos[i].get_field('ADZero')
            units[channel_row_map[i]] = stream.channel_infos[i].adc_step.units
            data[channel_row_map[i]] = ((data[channel_row_map[i]] - ad_zero)
                                        * adc_step)

        with open("assets/mcs_256mea_mapping.txt", "r") as ids_file:
            row_order = np.array([int(v) for v in ids_file.read().split(",") \
                    if v.strip() != ''])

        data = data[order]
        for i in [0, 15, 240, 255]:
            np.insert(data, i, np.nan, axis=0)
            np.insert(units, i, np.nan, axis=0)


        info = print_info(path, file_contents)
        data = Data(date, sampling_rate, units, data)
        del file_contents

    except IOError as err:
        info = "Failed to import specified file! Please specify a valid" \
                + " multi channel systems H5 formatted file.\n" \
                + "Error: " + str(err)
        data = None

    que.put((data, info))


def mcs_256mea_print_header_info(h5filename: str,
                                 data: McsPy.McsData.RawData
                                 ) -> str:
    """
    Prints infos that are contained in the header of the McS h5 file, like \
            the MEA name, the version, ...

        @param h5filename: Name of the file containing the data.
        @param data: McsPy.McData.RawData object

        @return the information formatted as a table
    """
    header_info = "\nFile path:" + h5filename + "\n\n"
    t_row = []
    delta = datetime.timedelta(microseconds=int(data.date_in_clr_ticks) / 10)
    date = datetime.datetime(1, 1, 1) + delta
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


def mcs_256mea_print_info(h5filename: str, data: McsPy.McsData.RawData) -> str:
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


def mcs_256mea_get_names() -> list[str]:
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


def mcs_256mea_get_name_row_map() -> dict[str, int]:
    """
    Returns a map from channel names to channel ids
    """
    names = get_mcs_256mea_names()
    ids = range(255)

    return dict(zip(names, ids))

"""
:author: Fabian Klopfer <fabian.klofper:ieee.org>
:date:   21.04.2023

Importer for multi channel systems 256MEAs chip, assuming all data is in a
single analog stream in the same recording.
"""
import datetime
from multiprocessing import Queue

import McsPy.McsData as Mcs256
import numpy as np
from tabulate import tabulate

from model.data import Recording


def mcs_256_import(path: str, que: Queue) -> None:
    """
    Import data recorded with a MultiChannel Systems MEA into A Data object, \
            see model/Data.py.

        :param path: the path to the file containing the data in McS h5 format.
        :param que: A python Queue, to return the loaded data asynchronously.

        :return a Data object containing the data in-memory and metadata or \
                None and an error message
    """
    fname = path.split('/')[-1].split('.')[0]

    Mcs256.VERBOSE = False
    try:
        file_contents = Mcs256.RawData(path)
        date = file_contents.date
        stream = file_contents.recordings[0].analog_streams[0]
        sampling_rate = stream.channel_infos[2].sampling_frequency.magnitude
        data = np.array(stream.channel_data, dtype=np.double)

        channel_row_map = {}

        for i in [c.channel_id for c in stream.channel_infos.values()]:
            channel_row_map[i] = stream.channel_infos[i].row_index

        # correct signal values, see MCS implementation:
        # https://mcspydatatools.readthedocs.io/en/latest/api.html#Mcs256.AnalogStream.get_channel_in_range
        for i in [c.channel_id for c in stream.channel_infos.values()]:
            adc_step = stream.channel_infos[i].adc_step.magnitude
            ad_zero = stream.channel_infos[i].get_field('ADZero')
            data[channel_row_map[i]] = ((data[channel_row_map[i]] - ad_zero)
                                        * adc_step)

        with (open("assets/mcs_256mea_mapping.txt", "r", encoding="utf-8")
                as ids_file):
            order = np.array([int(v) for v in ids_file.read().split(",")
                              if v.strip() != ''])

        data = data[order]

        n_mea_electrodes = 256
        side_len = int(np.sqrt(n_mea_electrodes))

        ground_els = np.array([0, side_len - 1, side_len * (side_len - 1),
                              side_len**2 - 1])

        names = np.array([f"R {i} C {j}" for i in range(1, side_len + 1)
                          for j in range(1, side_len + 1)])

        ground_el_names = names[ground_els]
        names = np.array([x for x in names if x not in ground_el_names])

        info = mcs_info(path, file_contents)
        rec = Recording(fname, date, n_mea_electrodes, sampling_rate, data, 0,
                        data.shape[1] - 1, names, ground_els, ground_el_names)
        del file_contents

    except IOError as err:
        info = "Failed to import specified file! Please specify a valid" \
                + " multi channel systems H5 formatted file.\n" \
                + "Error: " + str(err)
        rec = None

    # que.put((data, info))
    return rec, info


def mcs_header_info(h5filename: str,
                    data: Mcs256.RawData
                    ) -> str:
    """
    Prints infos that are contained in the header of the McS h5 file, like \
            the MEA name, the version, ...

        :param h5filename: Name of the file containing the data.
        :param data: McsPy.McData.RawData object

        :return the information formatted as a table
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


def mcs_info(h5filename: str, data: Mcs256.RawData) -> str:
    """
    Prints infos about the McS h5 file and the available stream(s)

        :param h5filename: Name of the file containing the data.
        :param data: : the object returned by calling McsPy.McData.RawData

        :return the information formatted as a table
    """
    info_string = mcs_header_info(h5filename, data) + "\n\n"
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

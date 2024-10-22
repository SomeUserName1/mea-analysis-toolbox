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
import McsPy.McsCMOSMEA as McsCMOS
import numpy as np
from tabulate import tabulate

from model.Data import Data


def mcs_cmos_import(path: str, que: Queue) -> None:
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

    McsCMOS.VERBOSE = False
    try:
        file_contents = McsCMOS.McsData(path)
        date = file_contents.attributes['DateTime']
        stream = file_contents.Acquisition.Sensor_Data
        num_channels = stream.SensorData_1_1.shape[-1] ** 2
        sampling_rate = int(np.round(stream.SensorMeta.Tick * 1e-6))
        print(stream.SensorMeta.Tick) # 50
        print(stream.SensorMeta.Label) # ROI
        print(stream.SensorMeta.RawDataType) # short
        print(stream.SensorMeta.Unit) # V
        print(stream.SensorMeta.Exponent) # -9
        print(stream.SensorMeta.ADCBits) # 16
        print(stream.SensorMeta['Conversion Factors']) # all arround 45000

        data = np.array(np.moveaxis(stream.SensorData_1_1, 0, -1), dtype=np.float64)
        data = data.reshape(num_channels, -1)
        print(data.shape)
        exit(0)

        channel_row_map = {}
        row_order = None

        for i in [c.channel_id for c in stream.channel_infos.values()]:
            row = stream.channel_infos[i].row_index
            channel_row_map[i] = row

        # correct signal values, see MCS implementation:
        # https://mcspydatatools.readthedocs.io/en/latest/api.html#McsCMOS.AnalogStream.get_channel_in_range
        for i in [c.channel_id for c in stream.channel_infos.values()]:
            adc_step = stream.channel_infos[i].adc_step.magnitude
            ad_zero = stream.channel_infos[i].get_field('ADZero')
            data[channel_row_map[i]] = ((data[channel_row_map[i]] - ad_zero)
                                        * adc_step)

        # TODO handle CMOS case and potentially others
        with open("assets/mcs_256mea_mapping.txt", "r") as ids_file:
            order = np.array([int(v) for v in ids_file.read().split(",") \
                    if v.strip() != ''])

        data = data[order]

        # TODO handle CMOS case and potentially others
        temp_data = np.concatenate((np.nan * np.ones((1, data.shape[1])), # 0
                                    data[0:14],                        # 1:14
                                    np.nan * np.ones((1, data.shape[1])), # 15
                                    data[14:238],                      # 14+2:238+1
                                    np.nan * np.ones((1, data.shape[1])), # 240
                                    data[238:],                        # 238+3:251+3
                                    np.nan * np.ones((1, data.shape[1])) # 255
                                    ), axis=0)
        grounds = [0, 15, 240, 255]
        data = temp_data
  #      info = mcs_info(path, file_contents)
        data = Data(date, sampling_rate, data, grounds)
        del file_contents

    except IOError as err:
        info = "Failed to import specified file! Please specify a valid" \
                + " multi channel systems H5 formatted file.\n" \
                + "Error: " + str(err)
        data = None

    que.put((data, info))


def mcs_header_info(h5filename: str,
                    data: McsCMOS.RawData
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


def mcs_info(h5filename: str, data: McsCMOS.RawData) -> str:
    """
    Prints infos about the McS h5 file and the available stream(s)

        @param h5filename: Name of the file containing the data.
        @param data: : the object returned by calling McsPy.McData.RawData

        @return the information formatted as a table
    """
    print(file_contents)
    print(stream)
     # print(stream.Acquisition_Info)
     # print(stream.Acquisition_Info.ChannelData_1)
     # print(stream.Acquisition_Info.ChannelMeta)
     # What is channel data?!
    print(stream.Sensor_Data)
    print(stream.Sensor_Data.SensorData_1_1)
    print(stream.Sensor_Data.SensorMeta)
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

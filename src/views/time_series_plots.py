"""
This module contains the function to plot the time series data in a grid.
"""
from multiprocessing import Process
import numpy as np
import pyqtgraph as pg

from model.data import Recording
from views.grid_plot_iterator import MEAGridPlotIterator


def plot_time_series_grid(rec: Recording,
                          selected: bool = False,
                          signals: bool = True,
                          peaks: bool = False,
                          bursts: bool = False,
                          seizure: bool = False,
                          thresh: bool = False):
    """
    Wrapper function to plot the time series data in a grid.
    Neccessary to run the plotting in a separate process before the callback
    exits.
    """
    proc = Process(target=do_plot, args=(rec, selected, signals,
                   peaks, bursts, seizure, thresh))
    proc.start()


def do_plot(rec: Recording,
            selected: bool,
            signals: bool,
            peaks: bool,
            bursts: bool,
            seizure: bool,
            thresh: bool):
    """
    Plot the time series data in a grid. The grid is created using the
    MEAGridPlotIterator class. The plots are created using the pyqtgraph
    library. The plots are added to a GraphicsLayoutWidget, which is then
    shown. The plots are linked in the y-direction. The plots are created
    using the data in the Data object. Only selected electrodes are plotted.
    The plots can be customized using the boolean arguments.

    :param rec: The recording object.
    :type rec: Recording

    :param selected: Indicated is the electrode selection was applied already.
    :type selected: bool

    :param signals: Plot the raw signals.
    :type signals: bool

    :param envelope: Plot the envelope of the signals.
    :type envelope: bool

    :param derivative: Plot the derivative of the signals.
    :type derivative: bool

    :param mv_average: Plot the moving average of the signals.
    :type mv_average: bool

    :param mv_mad: Plot the moving MAD of the signals.
    :type mv_mad: bool

    :param peaks: Plot the peaks of the signals.
    :type peaks: bool

    :param bursts: Plot the bursts of the signals.
    :type bursts: bool

    :param seizure: Plot the seizure of the signals.
    :type seizure: bool
    """
    sel_names = rec.get_sel_names()
    if len(sel_names) == 0:
        sel_names = rec.electrode_names

    t_start, t_stop = rec.get_time_s()
    data = rec.get_data()
    ts = np.linspace(t_start, t_stop, num=data.shape[1])
    sigs = (data[rec.selected_electrodes, rec.start_idx:rec.stop_idx]
            if not selected else data)
    mv_mads = rec.mv_mads.read() if thresh else None

    win = pg.GraphicsLayoutWidget(show=True, title="Raw signals")
    win.resize(1200, 800)
    prev_p = None
    it = MEAGridPlotIterator(rec)
    for i, (row, col) in enumerate(it):
        title_str = f'<font size="1">{sel_names[i]}</font>'
        p = win.addPlot(row=row, col=col, title=title_str)

        if signals:
            p.plot(x=ts, y=sigs[i], pen=(255, 255, 255, 200), label="Raw")

        if thresh:
            env_low = rec.envelopes[0][i]
            env_high = rec.envelopes[1][i]

            p.plot(x=ts[env_low], y=sigs[i][env_low], pen=(0, 255, 0, 255),
                   label="low envelope")

            p.plot(x=ts[env_high], y=sigs[i][env_high], pen=(0, 255, 0, 255),
                   label="high envelope")

            p.plot(x=ts, y=mv_mads[i], pen=(0, 255, 255, 255),
                   label="Moving MAD")

        if peaks:
            pdf = rec.peaks_df[rec.peaks_df['Channel'] == sel_names[i]]
            peak_idxs = pdf['PeakIndex'].values.astype(int)

            p.plot(x=ts[peak_idxs], y=sigs[i][peak_idxs], pen=None,
                   symbolBrush=(255, 0, 0, 255), symbolPen='w', label="Peaks")

            if thresh:
                p.plot(x=ts[rec.mad_env[i]], y=mv_mads[i][rec.mad_env[i]],
                       pen=(0, 255, 0, 255), label="moving MAD envelope")

                inf1 = pg.InfiniteLine(angle=0, pos=rec.lower[i],
                                       pen=(0, 0, 200),
                                       label="lower amplitude threshold")

                inf2 = pg.InfiniteLine(angle=0, pos=rec.upper[i],
                                       pen=(255, 0, 255),
                                       label="upper amplitude threshold")

                inf3 = pg.InfiniteLine(angle=0, pos=rec.mad_thresh[i],
                                       pen=(155, 165, 0),
                                       label="moving MAD threshold")
                p.addItem(inf1)
                p.addItem(inf2)
                p.addItem(inf3)
        if bursts:
            print("TODO")
            # TODO
        if seizure:
            print("TODO")
            # TODO

        p.setLabel('left', 'Voltag', units='V')
        p.setLabel('bottom', 'Time', unit='s')
        if prev_p is not None:
            p.setYLink(prev_p)

        prev_p = p

    pg.exec()

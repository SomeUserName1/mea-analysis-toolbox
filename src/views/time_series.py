from multiprocessing import Process

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from model.data import Data
from views.grid_plot_iterator import MEAGridPlotIterator


def do_plot(data: Data, signals, envelope, derivative, mv_average, mv_mad, mv_var, peaks, bursts, seizure):
    sel_names = data.get_sel_names()
    if len(sel_names) == 0:
        sel_names = data.electrode_names

    t_start, t_stop = data.get_time_s()
    ts = np.linspace(t_start, t_stop, num=data.data.shape[1])

    win = pg.GraphicsLayoutWidget(show=True, title="Raw signals")
    win.resize(800, 800)
    win.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
    win.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
    # pg.setConfigOptions(antialias=True)
    it = MEAGridPlotIterator(data)
    for i, (row, col) in enumerate(it):
        title_str = f'<font size="3">{sel_names[i]}</font>'
        p = win.addPlot(row=row, col=col, title=title_str)
        
        if signals:
            p.plot(x=ts, y=data.data[i])
        if envelope:
            p.plot(x=ts, y=data.envelopes[i])
        if derivative:
            p.plot(x=ts, y=data.derivatives[i])
        if mv_average:
            p.plot(x=ts, y=data.mv_means[i])
        if mv_mad:
            p.plot(x=ts, y=data.mv_mads[i])
        if mv_var:
            p.plot(x=ts, y=data.mv_var)
        if peaks:
            p.scatter(x=data.peaks[i][0], y=)# access peaks dict
        if bursts:
            # TODO
        if seizure:
            # TODO
        # FIXME continue here

        p.setLabel('left', units='V')
        p.setLabel('bottom', unit='s')


    pg.exec()

def plot_time_series_grid(data: Data):
    proc = Process(target=do_plot, args=(data,))
    proc.start()


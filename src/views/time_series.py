from multiprocessing import Process

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from model.data import Data
from views.grid_plot_iterator import MEAGridPlotIterator


def do_plot(data):
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
    for i, (row, col) in enumerate(MEAGridPlotIterator(data)):
        p = win.addPlot(x=ts, y=data.data[i], row=row, col=col) # title=sel_names[i], 
        if i == 0:
            p.setLabel('left', units='V')

    pg.exec()

def plot_time_series_grid(data: Data):
    proc = Process(target=do_plot, args=(data,))
    proc.start()


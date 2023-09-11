
"""
This module contains the function to plot the time series data in a grid.
"""
from multiprocessing import Process
import numpy as np
import pyqtgraph as pg

import sys
import pdb

from model.data import Recording
from views.grid_plot_iterator import MEAGridPlotIterator


def plot_psds_grid(rec: Recording):
    """

    """
    proc = Process(target=do_plot_psds, args=(rec,))
    proc.start()


def do_plot_psds(rec: Recording):
    """

    """
    sel_names = rec.get_sel_names()
    if len(sel_names) == 0:
        sel_names = rec.electrode_names

    t_start, t_stop = rec.get_time_s()
    freqs = rec.psds[0].read()
    pows = rec.psds[1].read()

    win = pg.GraphicsLayoutWidget(show=True, title="Raw signals")
    win.resize(1200, 800)
    prev_p = None
    it = MEAGridPlotIterator(rec)
    for i, (row, col) in enumerate(it):
        title_str = f'<font size="1">{sel_names[i]}</font>'
        p = win.addPlot(row=row, col=col, title=title_str)

        p.plot(x=freqs, y=pows[i])

        p.setLabel('left', 'Power', units='V^2/Hz')
        p.setLabel('bottom', 'Frequency', unit='Hz')
        if prev_p is not None:
            p.setYLink(prev_p)

        prev_p = p

    pg.exec()


def plot_spectrograms_grid(rec: Recording):
    """

    """
    proc = Process(target=do_plot_spectrograms, args=(rec,))
    proc.start()


def do_plot_spectrograms(rec: Recording):
    """

    """
    sel_names = rec.get_sel_names()
    if len(sel_names) == 0:
        sel_names = rec.electrode_names

    t_start, t_stop = rec.get_time_s()
    freqs = rec.spectrograms[0].read()
    times = rec.spectrograms[1].read()
    pows = rec.spectrograms[2].read()

    win = pg.GraphicsLayoutWidget(show=True, title="Raw signals")
    color_map = pg.colormap.get('viridis', source="matplotlib")
    pcs = []
    win.resize(1200, 800)
    prev_p = None
    max_pow = np.max(pows)
    min_pow = np.min(pows)
    sys.stdin = open(0)
    pdb.set_trace()
    it = MEAGridPlotIterator(rec)
    for i, (row, col) in enumerate(it):
        title_str = f'<font size="1">{sel_names[i]}</font>'
        p = win.addPlot(row=row, col=col, title=title_str)

        pc = pg.PColorMeshItem(x=times, y=freqs, z=pows[i], colorMap=color_map)
        pcs.append(pc)
        p.addItem(pc)

        p.setLabel('left', 'Frequency', unit='Hz')
        p.setLabel('bottom', 'Time', unit='s')
        if prev_p is not None:
            p.setYLink(prev_p)

        prev_p = p

    cbar = pg.ColorBarItem(values=(min_pow, max_pow), colorMap=color_map, label='Power')
    win.addItem(cbar)

    pg.exec()

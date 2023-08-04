import numpy as np
import pyqtgraph as pg
from multiprocessing import Process

from model.data import Data
from views.grid_plot_iterator import MEAGridPlotIterator


def do_plot(data):
    sel_names = data.get_sel_names()
    if len(sel_names) == 0:
        sel_names = data.electrode_names

    t_start, t_stop = data.get_time_s()
    ts = np.linspace(t_start, t_stop, num=data.data.shape[1])

    plot_app = pg.mkQApp("Raw Time Series Plot")
    win = pg.GraphicsLayoutWidget(show=True, title="Raw signals")
    win.resize(800, 800)
    pg.setConfigOptions(antialias=True)

    for i, (row, col) in enumerate(MEAGridPlotIterator(data)):
        print(f" idx {i}, row {row}, col {col}")
        p = win.addPlot(title=sel_names[i], x=ts, y=data.data[i], row=row, col=col)

    pg.exec()
    print("done")

def plot_time_series_grid(data: Data):
    proc = Process(target=do_plot, args=(data,))
    proc.start()
    proc.join()



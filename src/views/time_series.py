import numpy as np
import pyqtgraph as pg

from model.data import Data
from views.grid_plot_iterator import MEAGridPlotIterator

def plot_time_series_grid(data: Data):
    sel_names = data.get_sel_names()
    if len(sel_names) == 0:
        sel_names = data.electrode_names

    t_start, t_stop = data.get_time_s()
    ts = np.linspace(t_start, t_stop, num=data.data.shape[1])

    plot_app = pg.mkQApp("Raw Time Series Plot")
    layout = pg.LayoutWidget()
    for i, (row, col) in enumerate(MEAGridPlotIterator(data)):
        view = pg.widgets.RemoteGraphicsView.RemoteGraphicsView()
        plot_app.aboutToQuit.connect(view.close)
        layout.addWidget(view)

        p = view.pg.PlotItem(title=sel_names[i],)
        p._setProxyOptions(deferGetattr=True)
        view.setCentralItem(p)
        p.plot( x=ts, y=data.data[i], _callSync='off')


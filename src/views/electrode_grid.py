"""
Plots related to visualizations of the MEA electrode grid.
"""
from subprocess import Popen
import math
import os

from plotly import graph_objects as go
import numpy as np
from matplotlib import use, animation
import matplotlib.pyplot as plt

from constants import grid_size
from model.data import Recording
from views.grid_plot_utils import el_idx_data_to_plot, el_names_insert_grounds


def align_image(x: int, y: int, sizex: int, sizey: int, sizing: str) -> None:
    """
    Stub to be used to align the image in the selection screen to the \
            electrode grid when the image is set as background \
            (not implemented yet).
    """
    # fig2.update_layout(images=[
    #     dict(source=img, xref="paper", yref="paper", x=0, y=1, sizex=1,
    #         sizey=1, sizing="stretch", opacity=0.5, layer="above")])
    pass


def get_marked_coords(rec: Recording) -> tuple[np.ndarray,
                                               np.ndarray,
                                               np.ndarray,
                                               np.ndarray,
                                               list[str],
                                               list[str]]:
    """
    Get the coordinates of the selected electrodes in the grid. The grid is
    assumed to be square. The coordinates are returned as flattened arrays.
    The coordinates of the unselected electrodes are returned as flattened
    arrays. The names of the selected electrodes are returned as a list of
    strings. The names of the unselected electrodes are returned as a list of
    strings. The coordinates are returned in the order of the electrode names.

    :param rec: The recording object.
    :type rec: Recording

    :return: The coordinates of the selected electrodes in the grid, the
             coordinates of the unselected electrodes in the grid, the names of
              the selected electrodes, the names of the unselected electrodes.
              The coordinates are returned as flattened arrays, i.e. the first
              array contains all x coordinates, the second array contains all y
              of the selected electrodes each. The third and fourth array
              contain the x and y coordinates of the unselected electrodes.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str],
                  list[str]]
    """
    # generate the grid
    side_len = int(math.sqrt(rec.n_mea_electrodes))
    x_label = np.linspace(1, side_len, side_len)
    xx_all, yy_all = np.meshgrid(x_label, x_label, sparse=False, indexing='xy')
    xx_all = xx_all.flatten()
    yy_all = yy_all.flatten()

    sel = el_idx_data_to_plot(rec)
    names_all = el_names_insert_grounds(rec)
    unsel = [x for x in range(rec.n_mea_electrodes) if x not in sel]

    names_sel = [names_all[i] for i in sel]
    xx = xx_all[sel]
    yy = yy_all[sel]
    names_un = [names_all[i] for i in unsel]
    xx_un = xx_all[unsel]
    yy_un = yy_all[unsel]

    return xx, yy, xx_un, yy_un, names_sel, names_un


def draw_electrode_grid(rec: Recording) -> go.Figure:
    """
    Draws the electrode grid with the selected electrodes colored green and the
    unselected electrodes colored red. The grid is assumed to be square.

    :param rec: The recording object.
    :type rec: Recording

    :return: The plotly figure.
    :rtype: go.Figure
    """
    x_sel, y_sel, x_un, y_un, names_sel, names_un = get_marked_coords(rec)

    grid = go.Figure()
    grid.add_trace(
        go.Scatter(x=x_un, y=y_un, marker={'color': 'red',
                                           'showscale': False},
                   mode='markers', name='Unselected Electrodes',
                   hovertemplate='<b>%{text}</b>',
                   text=names_un,
                   showlegend=False))

    grid.add_trace(
        go.Scatter(x=x_sel, y=y_sel, marker={'color': 'green',
                                             'showscale': False},
                   mode='markers', name='Selected Electrodes',
                   hovertemplate='<b>%{text}</b>',
                   text=names_sel,
                   showlegend=False))

    grid.update_xaxes(showline=True, linewidth=1, linecolor='black',
                      range=[0, np.sqrt(rec.n_mea_electrodes) + 1],
                      mirror=True)

    grid.update_yaxes(showline=True, linewidth=1, linecolor='black',
                      mirror=True,
                      range=[0, np.sqrt(rec.n_mea_electrodes) + 1],
                      autorange="reversed")

    grid.update_layout(template="plotly_white", width=grid_size,
                       height=grid_size, legend=dict(orientation="h"))

    return grid


# FIXME use pyqtgraph
def create_video(rec: Recording, bins: np.ndarray, fps: float) -> None:
    """
    Creates a video animation of the electrode grid using matplotlib and \
            ffmpeg, given a properly binned version of the data, the fps and \
            the time window. Opens and plays that video using vlc

        :param rec: Recording object used to align the grids to the data rows.
        :param bins: binned data
        :param fps: the frame rate
    """
    use('AGG')
    base_path = os.path.join(os.getcwd(), "plots")
    video_name = os.path.join(base_path, "amplitude-animation.mp4")

    xx, yy, xx_un, yy_un, _, _ = get_marked_coords(rec)
    lims = [0, np.amax(bins)]

    fig = plt.figure()
    writer = animation.FFMpegWriter(
                 fps=float(fps), bitrate=-1)

    with writer.saving(fig, video_name, dpi=100):
        for i in range(bins.shape[0]):
            img = plt.scatter(xx, yy, c=bins[i], cmap='YlOrRd', vmin=lims[0],
                              vmax=lims[1])
            plt.scatter(xx_un, yy_un, c='grey')
            cbar = plt.colorbar(img)
            cbar.ax.set_ylabel(r'Average Amplitude [$\mu$V]', rotation=270,
                               labelpad=50)
            plt.yticks([])
            plt.xticks([])
            plt.gca().invert_yaxis()

            writer.grab_frame()
            plt.clf()

    writer.finish()

    _ = Popen(['vlc', video_name])


# FIXME use pyqtgraph
def plot_value_grid(rec: Recording, values: np.ndarray) -> None:
    """
    Plots an electrode grid and colors the points based on the values.
    Can be used to display e.g. the RMS graphically instead of as a table.

        :param data: data Object, see model/Data.py
        :param values: a 1D array with exactly 252 values aligned to the data \
                rows
    """
    xx, yy, xx_un, yy_un, names_sel, _ = get_marked_coords(rec)
    lims = [np.amin(values), np.amax(values)]

    _ = plt.figure()
    plt.scatter(xx, yy, c=values, cmap='seismic', vmin=lims[0], vmax=lims[1])
    plt.scatter(xx_un, yy_un, c='grey')
    plt.show()

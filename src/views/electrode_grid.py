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
from model.data import Data

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


def get_marked_coords(data: Data) -> tuple[np.ndarray, # x values for selected electr.
                                  np.ndarray, # y values for selected electr.
                                  np.ndarray, # x values for unsel electr.
                                  np.ndarray, # y values for unsel electr.
                                  list[str], # names of selected el.
                                  list[str]]: # limits of array
    """
    returns the coordinates in the grid of each electrode for selected and
    unselected electrodes each, along with the names of the selected electrodes.

    @param data: data object, see model/Data.py

    @return: array of x coordinates and y coordinates for selected and
    unselected electrodes each, along with the names each.
    """
    # generate the grid
    side_len = int(math.sqrt(data.n_mea_electrodes))
    x_label = np.linspace(1, side_len, side_len)
    xx_all, yy_all = np.meshgrid(x_label, x_label, sparse=False, indexing='xy')
    xx_all = xx_all.flatten()
    yy_all = yy_all.flatten()

    sel = data.selected_electrodes.copy()
    names_all = data.electrode_names.tolist()
    for i in range(data.ground_els.shape[0]):
        names_all.insert(data.ground_els[i], data.ground_el_names[i])

    # Convert the selected electrode indexes from conforming to the data matrix
    # to a row-major enumeration __including__ the ground electrodes for plotting
    for i, idx in enumerate(sel):
        if idx >= data.ground_els[2]:
            sel[i] += 3
        elif idx >= data.ground_els[1]:
            sel[i] += 2
        elif idx >= data.ground_els[0]:
            sel[i] += 1
    # FIXME: adjacent to ground el selection off by one, plots only one electrode at a time, when mutliple are selected
    unsel = [x for x in range(data.n_mea_electrodes) if x not in sel]

    names_sel = [names_all[i] for i in sel]
    xx = xx_all[sel]
    yy = yy_all[sel]
    names_un = [names_all[i] for i in unsel]
    xx_un = xx_all[unsel]
    yy_un = yy_all[unsel]



    return xx, yy, xx_un, yy_un, names_sel, names_un


def draw_electrode_grid(data: Data) -> go.Figure:
    """
    MEA electrode grid visualization using the Dash/Plotly framework, i.e. \
            in the browser.

    Returns:
        Plotly figure with n_electrodes electrodes rendered as scatter plot.

    """
    x_sel, y_sel, x_un, y_un, names_sel, names_un = get_marked_coords(data)

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
                      range=[0, np.sqrt(data.n_mea_electrodes) + 1],
                      mirror=True)

    grid.update_yaxes(showline=True, linewidth=1, linecolor='black',
                      mirror=True,
                      range=[0, np.sqrt(data.n_mea_electrodes) + 1],
                      autorange="reversed")

    grid.update_layout(template="plotly_white", width=grid_size,
                       height=grid_size, legend=dict(orientation="h"))

    return grid


def create_video(data: Data, bins: np.ndarray, fps: float) -> None:
    """
    Creates a video animation of the electrode grid using matplotlib and \
            ffmpeg, given a properly binned version of the data, the fps and \
            the time window. Opens and plays that video using vlc

        @param data: Data object used to align the grids to the data rows.
        @param bins: binned data
        @param fps: the frame rate
    """
    use('AGG')
    base_path = os.path.join(os.getcwd(), "plots")
    video_name = os.path.join(base_path, "amplitude-animation.mp4")

    xx, yy, xx_un, yy_un, _, _ = get_marked_coords(data)
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


def plot_value_grid(data: Data, values: np.ndarray) -> None:
    """
    Plots an electrode grid and colors the points based on the values.
    Can be used to display e.g. the RMS graphically instead of as a table.

        @param data: data Object, see model/Data.py
        @param values: a 1D array with exactly 252 values aligned to the data \
                rows
    """
    xx, yy, xx_un, yy_un, names_sel, _ = get_marked_coords(data)
    lims = [np.amin(data.data), np.amax(data.data)]

    _ = plt.figure()
    plt.scatter(xx, yy, c=values, cmap='seismic', vmin=lims[0], vmax=lims[1])
    plt.scatter(xx_un, yy_un, c='grey')
    plt.show()

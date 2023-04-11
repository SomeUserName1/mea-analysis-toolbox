"""
Plots related to visualizations of the MEA electrode grid.
"""
from subprocess import Popen
import os

from plotly import graph_objects as go
import numpy as np
from matplotlib import use, animation
import matplotlib.pyplot as plt

from model.io.import_mcs import get_mcs_256mea_row_name_dict, \
        get_mcs_256mea_row_order

grid_size = 500

def align_image(x, y, sizex, sizey, sizing):
    """
    Stub to be used to align the image in the selection screen to the \
            electrode grid when the image is set as background \
            (not implemented yet).
    """
    # fig2.update_layout(images=[
    #     dict(source=img, xref="paper", yref="paper", x=0, y=1, sizex=1,
    #         sizey=1, sizing="stretch", opacity=0.5, layer="above")])
    pass


def align_grid_to_rows(data):
    """
    returns the coordinates in the grid of each electrode for both selected \
            and unselected electrodes, along with the names of the selected \
            electrodes and the min and max value of the data.

        @param data: data object, see model/Data.py

        @return: array of x coordinates and y coordinates for both selected \
                and unselected electrodes, along with the names of the \
                selected electrodes and the min and max values of the data.
    """
    # generate the grid
    x_label = np.linspace(1, 16, 16)
    xx_all, yy_all = np.meshgrid(x_label, x_label, sparse=False, indexing='xy')
    xx_all = np.delete(xx_all.flatten(), [0, 15, 240, 255])
    yy_all = np.delete(yy_all.flatten(), [0, 15, 240, 255])
    names = get_mcs_256mea_row_name_dict()
    order = get_mcs_256mea_row_order()
    lims = [np.amin(data.data), np.amax(data.data)]
    unsel = [x for x in range(data.num_electrodes) \
                         if x not in data.selected_rows]

    xx = []
    yy = []
    xx_un = []
    yy_un = []
    names_sel = []
    names_un = []
    # reorder it such that x[i], y[i] correspond to data[i]
    for idx in range(data.num_electrodes):
        coord_idx = np.where(order == idx)[0][0]
        if idx in data.selected_rows:
            xx.append(xx_all[coord_idx])
            yy.append(yy_all[coord_idx])
            names_sel.append(names[idx])
        elif idx in unsel:
            names_un.append(names[idx])
            xx_un.append(xx_all[coord_idx])
            yy_un.append(yy_all[coord_idx])
        else:
            raise RuntimeError(f"Found a channel that is neither selected"
                               " not unselected. Row ID: {idx}")

    return np.array(xx), np.array(yy), np.array(xx_un), np.array(yy_un), \
            names_sel, names_un, lims


def draw_electrode_grid(n_electrodes_per_side, x_selected, y_selected,
        names_selected, x_unselected, y_unselected, names_unselected):
    """
    MEA electrode grid visualization using the Dash/plotly framework, i.e. \
            in the browser.

    Returns:
        Plotly figure with n_electrodes electrodes rendered as scatter plot.

    """
    grid = go.Figure()
    grid.add_trace(
        go.Scatter(x=x_unselected, y=y_unselected, marker={'color': 'red',
                                             'showscale': False},
                   mode='markers', name='Unselected Electrodes',
                   hovertemplate='<b>%{text}</b>',
                   text=names_unselected,
                   showlegend=False))

    grid.add_trace(
        go.Scatter(x=x_selected, y=y_selected, marker={'color': 'green',
                                             'showscale': False},
                   mode='markers', name='Selected Electrodes',
                   hovertemplate='<b>%{text}</b>',
                   text=names_selected,
                   showlegend=False))

    grid.update_xaxes(showline=True, linewidth=1, linecolor='black',
                      range=[0, n_electrodes_per_side + 1], mirror=True)

    grid.update_yaxes(showline=True, linewidth=1, linecolor='black',
                      mirror=True, range=[0, n_electrodes_per_side + 1],
                      autorange="reversed")

    grid.update_layout(template="plotly_white", width=grid_size,
                       height=grid_size, legend=dict(orientation="h"))

    return grid


def create_video(data, bins, fps):
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

    xx, yy, xx_un, yy_un, _, _, _ = align_grid_to_rows(data)
    lims = [0, np.amax(bins)]

    fig = plt.figure()
    writer = animation.FFMpegWriter(
                 fps=float(fps), bitrate=-1)

    with writer.saving(fig, video_name, dpi=100):
        for i in range(bins.shape[0]):
            img = plt.scatter(xx, yy, c=bins[i], cmap='YlOrRd', vmin=lims[0], vmax=lims[1])
            plt.scatter(xx_un, yy_un, c='grey')
            cbar = plt.colorbar(img)
            cbar.ax.set_ylabel(r'Average Amplitude [$\mu$V]', rotation=270, labelpad=50)
            plt.yticks([])
            plt.xticks([])
            plt.gca().invert_yaxis()

            writer.grab_frame()
            plt.clf()

    writer.finish()

    _ = Popen(['vlc', video_name])


def plot_value_grid(data, values):
    """
    Plots an electrode grid and colors the points based on the values. 
    Can be used to display e.g. the RMS graphically instead of as a table.

        @param data: data Object, see model/Data.py
        @param values: a 1D array with exactly 252 values aligned to the data \
                rows
    """
    xx, yy, xx_un, yy_un, names_sel, _, lims = align_grid_to_rows(data)

    _ = plt.figure()
    plt.scatter(xx, yy, c=values, cmap='seismic', vmin=lims[0], vmax=lims[1])
    plt.scatter(xx_un, yy_un, c='grey')
    plt.show()

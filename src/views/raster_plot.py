import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from model.io.import_mcs import get_mcs_256mea_row_order

def plot_raster(spikes, electrode_idxs, fs, names, t_start, t_end):
    if t_start is None or t_end is None:
        raise RuntimeError("You must specify the start and end point in time in mu s!")

    mpl.use('TkAgg')

    t_start_idx = int(np.round(fs * t_start / 1000000))
    t_end_idx = int(np.round(fs * t_end / 1000000))
    n_samples = t_end_idx - t_start_idx

    time = np.linspace(t_start / 1000000, t_end / 1000000, n_samples)

    fig = plt.figure()
    ax = plt.subplot(111)

    max_h = 0
    for i in range(len(electrode_idxs)):
        for j in range(spikes[i][1].shape[0]):
            if spikes[i][1][j] > max_h:
                max_h = spikes[i][1][j]
                
    row_order = get_mcs_256mea_row_order()
    row_order = [row for row in row_order if row in electrode_idxs]

    norm = mpl.colors.Normalize(vmin=0.0, vmax=max_h)
    cmap = plt.get_cmap('Reds', int(np.ceil(max_h)))
    sel_names = []
    for i in range(len(electrode_idxs)):
        row = electrode_idxs.index(row_order[i])
        sel_names.append(names[row])
        spike_times = [time[int(j)] for j in spikes[row][0]]
        plt.vlines(spike_times, i, i+1)
        lines = plt.gca().collections[i]
        colors = lines.get_colors()
        num_colors = len(colors)
        num_lines = len(lines.get_paths())
        new_colors = np.tile(colors, ((num_lines + num_colors - 1) // num_colors, 1))

        for j in range(num_lines):
            new_colors[j] = cmap(norm(spikes[row][1][j]))

        lines.set_colors(new_colors)

    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Electrode')
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim(0, len(electrode_idxs))
    plt.yticks([x+0.5 for x in range(len(electrode_idxs))], sel_names)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax)
    cbar.ax.set_ylabel(r'Peak Amplitude [$\mu$V]', rotation=270, labelpad=50)
    plt.show()
 

def plot_psth(spikes, electrode_idxs, fs, t_start, t_end):
    if t_start is None or t_end is None:
        raise RuntimeError("You must specify the start and end point in time in mu s!")

    mpl.use('TkAgg')

    fig = plt.figure()
    ax=plt.subplot(111)

    t_tot_ms = int(np.ceil((t_end - t_start) / 10000))
    bins = np.zeros(t_tot_ms)
    sum_hs = 0

    for i in range(len(electrode_idxs)):
        for j in range(spikes[i][0].shape[0]):
            bins[int(spikes[i][0][j] / fs * 100)] += spikes[i][1][j]
            sum_hs += spikes[i][1][j]

    time = np.linspace(t_start / 1000000, t_end / 1000000, t_tot_ms)
    bins = bins / fs * 100 / len(electrode_idxs)
    ax.bar(time, bins)

    ax.set_xlim(time[0], time[-1])
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel(r'Average Amplitude [$\mu$V]')
    plt.tight_layout()
    plt.show()


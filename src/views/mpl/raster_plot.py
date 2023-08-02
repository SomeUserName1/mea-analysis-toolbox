import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def plot_raster(spikes, data):
    mpl.use('TkAgg')

    electrode_idxs = data.selected_rows
    n_samples = data.stop_idx - data.start_idx
    t_start = data.start_idx / sampling_rate
    t_stop = data.start_idx / sampling_rate
    time = np.linspace(t_start, t_stop, n_samples)

    max_h = 0
    for i in range(len(electrode_idxs)):
        for j in range(spikes[i][1].shape[0]):
            if spikes[i][1][j] > max_h:
                max_h = spikes[i][1][j]
                
    fig = plt.figure()
    ax = plt.subplot(111)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=max_h)
    cmap = plt.get_cmap('Reds', int(np.ceil(max_h)))
    sel_names = data.get_selected_names()

    for i in range(len(electrode_idxs)):
        row = electrode_idxs.index(i)
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
    ax.set_xlim(t_start, t_stop)
    ax.set_ylim(0, len(electrode_idxs))
    plt.yticks([x+0.5 for x in range(len(electrode_idxs))], sel_names)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax)
    cbar.ax.set_ylabel(r'Peak Amplitude [$\mu$V]', rotation=270, labelpad=50)
    plt.show()
 

def plot_psth(spikes, data): # electrode_idxs, fs, t_start, t_end):
    mpl.use('TkAgg')

    fs = data.sampling_rate
    electrode_idxs = data.selected_rows
    t_start = data.start_idx / sampling_rate * 1000
    t_stop = data.start_idx / sampling_rate * 1000
    tot_ms = int(np.ceil((t_stop - t_start)))
    time = np.linspace(t_start, t_stop, tot_ms)

    bins = np.zeros(tot_ms)
    for i in range(len(electrode_idxs)):
        for j in range(spikes[i][0].shape[0]):
            bins[int(spikes[i][0][j] / fs * 1000)] += spikes[i][1][j]

    bins = bins / fs * 1000 / len(electrode_idxs)

    fig = plt.figure()
    ax=plt.subplot(111)
    ax.bar(time, bins)
    ax.set_xlim(time[0], time[-1])
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel(r'Average Amplitude [$\mu$V]')
    plt.tight_layout()
    plt.show()


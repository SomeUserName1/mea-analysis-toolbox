from datetime import datetime
import os
from multiprocessing import Process

from matplotlib import use
import matplotlib.pyplot as plt
import numpy as np

from model.io.import_mcs import get_mcs_256mea_row_order


def plot_in_grid(kind, signals, electrode_idxs, names, fs=None, t_start=None, t_end=None, double=None, scatter=None, burst_idxs_all=None, burst_idxs_longest=None):
    use('TkAgg')

    if kind == 'time_series':
        t_start_idx = int(np.round(fs * t_start / 1000000))
        t_end_idx = int(np.round(fs * t_end / 1000000))
        n_samples = t_end_idx - t_start_idx + 1
        time = np.linspace(t_start / 1000000, t_end / 1000000, signals[0].shape[0])
        y_range = [np.amin(signals), np.amax(signals)]

        if double is not None:
            y_range = [min(y_range[0], np.amin(double)), max(y_range[1], np.amax(double))]

    elif kind == 'spectrograms':
        max_pow = 0
        for spect in signals[0]:
            if np.amax(spect) > max_pow:
                max_pow = np.amax(spect)


    row_order = get_mcs_256mea_row_order()
    grid_sz = 16

    offset = 0
    plotted = np.zeros((grid_sz, grid_sz))
    for i in range(16):
        if i * grid_sz >= 252:
            break
        for j in range(16):
            if i * grid_sz + j >= 252:
                break

            if i * grid_sz + j == 0 \
                    or i * grid_sz + j == grid_sz - 1 \
                    or i * grid_sz + j == grid_sz * (grid_sz - 1) \
                    or i * grid_sz + j == grid_sz * grid_sz - 1:
                offset += 1
                if i * grid_sz + j == 0 and 'B1' in names and 'A2' in names \
                    or i * grid_sz + j == grid_sz - 1 and 'P1' in names and 'R2' in names \
                    or i * grid_sz + j == grid_sz * (grid_sz - 1) and 'A15' in names and 'B16' in names \
                    or i * grid_sz + j == grid_sz * grid_sz - 1 and 'R15' in names and 'R16' in names:
                    plotted[i, j] = 1
                continue

            if row_order[i * grid_sz + j - offset] in electrode_idxs:
                plotted[i, j] = 1

    empty_rows = []
    empty_cols = []
    for idx in range(grid_sz):
        if np.sum(plotted[idx, :]) == 0:
            empty_rows.append(idx)

        if np.sum(plotted[:, idx]) == 0:
            empty_cols.append(idx)

    grid_y = grid_sz - len(empty_rows)
    grid_x = grid_sz - len(empty_cols)
    fig, ax_array = plt.subplots(grid_y, grid_x, sharey=True, sharex=True)


    if grid_y == 1:
        ax_array = np.expand_dims(ax_array, axis=0)
    if grid_x == 1:
        ax_array = np.expand_dims(ax_array, axis=1)

    
    row_offset = 0
    col_offset = 0
    offset = 0
    cont = False
    for i in range(grid_sz):
        col_offset = 0
        if i in empty_rows:
            row_offset += 1

            if i == 0:
                offset += 2

            continue

        for j in range(grid_sz):
            if i * grid_sz + j == 0\
                    or i * grid_sz + j == grid_sz - 1\
                    or i * grid_sz + j == grid_sz * (grid_sz - 1)\
                    or i * grid_sz + j == grid_sz * grid_sz - 1:
                offset += 1
                cont = True

            if j in empty_cols:
                col_offset += 1
                cont = True

            if cont:
                cont = False
                continue

            if row_order[i * grid_sz + j - offset] not in electrode_idxs:
                continue

            ax_i = i - row_offset
            ax_j = j - col_offset

            row = electrode_idxs.index(row_order[i * grid_sz + j - offset])
            
            if kind == 'time_series':
                ax_array[ax_i, ax_j].plot(time, signals[row], label=names[row], zorder=0)
                ax_array[ax_i, ax_j].set_ylim(bottom=y_range[0], top=y_range[1])

                if len(electrode_idxs) < 252:
                    ax_array[ax_i, ax_j].legend()

                if ax_i == grid_y - 1 and ax_j == 0:
                    ax_array[ax_i, ax_j].set_xlabel('Time [sec]')
                    ax_array[ax_i, ax_j].set_ylabel(r'Amplitude [$\mu$V]')

                if double is not None:
                    ax_array[ax_i, ax_j].plot(time, double[row], zorder=3)

                if scatter is not None:
                    scatter_time = [time[int(i)] for i in scatter[row][0]]
                    ax_array[ax_i, ax_j].scatter(scatter_time, scatter[row][1], c='red', zorder=4)

                if burst_idxs_longest is not None and len(burst_idxs_longest[row]) > 0:
                    start_idx = burst_idxs_longest[row][0]
                    stop_idx = burst_idxs_longest[row][1]
                    ax_array[ax_i, ax_j].plot(time[start_idx : stop_idx], signals[row][start_idx : stop_idx], c='red', zorder=2)

                if burst_idxs_all is not None:
                    for k in range(burst_idxs_all[row][0].shape[0]):
                        start_idx = burst_idxs_all[row][0][k]
                        end_idx = burst_idxs_all[row][1][k]
                        ax_array[ax_i, ax_j].plot(time[start_idx : end_idx], signals[row][start_idx : end_idx], c='green', zorder=1)

            elif kind == 'psds':
                ax_array[ax_i, ax_j].semilogy(signals[row, 0], signals[row, 1], label=names[row])

                if len(electrode_idxs) < 252:
                    ax_array[ax_i, ax_j].legend(loc="lower left")

                if ax_i == grid_y - 1 and ax_j == 0:
                     ax_array[ax_i, ax_j].set_xlabel('Frequency [Hz]')
                     ax_array[ax_i, ax_j].set_ylabel(r'PSD [$\mu V^2$/Hz]')

                ax_array[ax_i, ax_j].axvline(0.5, color="grey")
                ax_array[ax_i, ax_j].axvline(4, color="grey")
                ax_array[ax_i, ax_j].axvline(8, color="grey")
                ax_array[ax_i, ax_j].axvline(13, color="grey")
                ax_array[ax_i, ax_j].axvline(30, color="grey")
                ax_array[ax_i, ax_j].axvline(90, color="grey")
                ax_array[ax_i, ax_j].text(1, 1, r"$\delta$")
                ax_array[ax_i, ax_j].text(5, 1, r"$\theta$")
                ax_array[ax_i, ax_j].text(9, 1, r"$\alpha$")
                ax_array[ax_i, ax_j].text(14, 1, r"$\beta$")
                ax_array[ax_i, ax_j].text(36, 1, r"$\gamma$")            

            elif kind == 'spectrograms':
                img = ax_array[ax_i, ax_j].pcolormesh(signals[1][row], signals[0][row], signals[2][row], cmap='hot', vmax=max_pow, rasterized=True)
                ax_array[ax_i, ax_j].set_title(names[row])
                ax_array[ax_i, ax_j].set_ylim([0, 500])

                if ax_i == grid_y - 1 and ax_j == 0:
                    ax_array[ax_i, ax_j].set_ylabel('Frequency [Hz]')
                    ax_array[ax_i, ax_j].set_xlabel('Time [sec]')

            elif kind == 'periodic_aperiodic':
                signals[row].plot(plot_aperiodic=True, plot_peaks='shade', plt_log=False, ax=ax_array[ax_i, ax_j])

                if len(electrode_idxs) < 252:
                    ax_array[ax_i, ax_j].legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left")

            else:
                raise RuntimeError("Invalid plot kind!")

    if kind == 'spectrograms':
        cbar = fig.colorbar(img, ax=ax_array.ravel().tolist())
        cbar.ax.set_ylabel(r'PSD [$V^2$/Hz]', rotation=90, labelpad=50)

    plt.show()
    plt.close()

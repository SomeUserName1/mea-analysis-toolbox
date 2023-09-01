import numpy as np

from model.data import Recording


# Convert the selected electrode indexes from conforming to the data matrix
# to a row-major enumeration __including__ the ground electrodes for plotting
def el_idx_data_to_plot(rec: Recording) -> np.ndarray:
    sel = rec.selected_electrodes.copy()
    for i, idx in enumerate(sel):
        if idx >= rec.ground_els[2] - 2:
            sel[i] += 3
        elif idx >= rec.ground_els[1] - 1:
            sel[i] += 2
        elif idx >= rec.ground_els[0]:
            sel[i] += 1

    return sel


def el_idx_plot_to_data(rec: Recording, idx: int) -> int:
    if idx in rec.ground_els:
        idx = -1
    else:
        # The grid also shows the ground electrodes.
        # the indexes must be subtracted accordingly to
        # be fitting to the data matrix
        if idx >= rec.ground_els[2]:
            idx -= 3
        elif idx >= rec.ground_els[1]:
            idx -= 2
        elif idx >= rec.ground_els[0]:
            idx -= 1

    return idx


def el_names_insert_grounds(rec: Recording) -> np.ndarray:
    names_all = rec.electrode_names.tolist()
    for i in range(rec.ground_els.shape[0]):
        names_all.insert(rec.ground_els[i], rec.ground_el_names[i])

    return np.array(names_all)


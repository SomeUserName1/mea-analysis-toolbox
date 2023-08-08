import numpy as np

from model.data import Data

# Convert the selected electrode indexes from conforming to the data matrix
# to a row-major enumeration __including__ the ground electrodes for plotting
def el_idx_data_to_plot(data: Data) -> np.ndarray:
    sel = data.selected_electrodes.copy()
    for i, idx in enumerate(sel):
        if idx >= data.ground_els[2] - 2:
            sel[i] += 3
        elif idx >= data.ground_els[1] - 1:
            sel[i] += 2
        elif idx >= data.ground_els[0]:
            sel[i] += 1

    return sel


def el_idx_plot_to_data(data: Data, idx: int) -> int:
    if idx in data.ground_els:
        idx = -1 
    else:
        # The grid also shows the ground electrodes.
        # the indexes must be subtracted accordingly to 
        # be fitting to the data matrix
        if idx >= data.ground_els[2]:
            idx -= 3
        elif idx >= data.ground_els[1]:
            idx -= 2
        elif idx >= data.ground_els[0]:
            idx -= 1

    return idx


def el_names_insert_grounds(data: Data) -> np.ndarray:
    names_all = data.electrode_names.tolist()
    for i in range(data.ground_els.shape[0]):
        names_all.insert(data.ground_els[i], data.ground_el_names[i])
 
    return np.array(names_all)


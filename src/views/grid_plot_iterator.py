from model.data import Data

def MEAGridPlotIterator(data: Data):
    """
    An iterator class that iterates through channels to be plotted on a grid.
    """
    def __init__(self,
                 patient: Patient) -> None:
        """
        Initialize the MEAGridPlotIterator object with a patient object.

        Parameters
        ----------
        data: model.data.Data
           The Data object containing channel data, selected channels
           and metadata to be plotted.
        """
        self.data = data
        self.axs = None
        self.fig = None
        self.data_idx 
        self.row_idx = 0
        self.col_idx = 0
        # See which electrodes are selected and what has to be plotted, such
        # that the grid is as small as possible. Also check if corners are
        # contained, to draw them if neccessary.
        grid_sz = np.sqrt(data.num_electrodes)
        selected = data.selected_rows
        offset = 0
        crnrs = [0, grid_sz - 1, grid_sz * (grid_sz - 1), grid_size ** 2 - 1]
        plotted = np.zeros((grid_sz, grid_sz))
        for i in range(grid_sz):
            for j in range(grid_sz):
                idx = i * grid_sz + j
                if idx >= grid_size * grid_size:
                    break
                # if the channel is a corner and adjacent channels are selected
                # mark it as to be plotted
                if idx == crnrs[0] and crnrs[0] + 1 in selected \
                        and crnrs[0] + grid_sz in selected \
                    or idx == crnrs[1] and crnrs[1] - 1 in selected \
                        and crnrs[1] + grid_sz in selected \
                    or idx == crnrs[2] and crnrs[2] + 1 in selected \
                        and crnrs[2] - grid_sz in selected \
                    or idx == crnrs[3] and crnrs[3] - 1 in selected \
                        and crnrs[3] - grid_sz in selected:
                        plotted[i, j] = 1
                        continue
                # If the channel is selected, mark it as to be plotted
                if idx in electrode_idxs:
                    plotted[i, j] = 1

        # check for empty rows and columns.
        # if its the first row/col print it st. the cut in the grid is visible
        # for every following empty row/col omit it from the grid, st. the
        # grid is as small as possible
        self.empty_rows = []
        self.empty_cols = []
        prev_r_empty = False
        prev_c_empty = False
        for idx in range(grid_sz):
            if np.sum(plotted[idx, :]) == 0:
                if prev_r_empty:
                    empty_rows.append(idx)

                prev_r_empty = True
            else:
                prev_r_empty = False

            if np.sum(plotted[:, idx]) == 0:
                if prev_c_empty:
                    empty_cols.append(idx)

                prev_c_empty = True
            else:
                prev_c_empty = False

        # adjust the grid size according to the empty rows and cols
        # and create the figure and axes objects
        grid_y = grid_sz - len(empty_rows)
        grid_x = grid_sz - len(empty_cols)
        self.fig, self.axs = plt.subplots(grid_y, grid_x, sharey=True,
                                          sharex=True)
        # Matplotlib returns a single axes object if there is only one row or
        # column. To make the indexing work, we have to make sure that the
        # axes object is always a 2D array.
        if grid_y == 1:
            self.axs = np.expand_dims(self.axs, axis=0)
        if grid_x == 1:
            self.axs = np.expand_dims(self.axs, axis=1)


    def __iter__(self):
        """
        Implement the iterator protocol.

        Returns
        -------
        self: GridPlotIterator
        """
        return self


    def __next__(self) -> plt.Axes, np.ndarray:
        """
        Get the next axes of the grid plot.

        Returns
        -------
        data : plt.Axes
            The axes to be plotted to next according to the electrode on the MEA
            grid.

        Raises
        ------
        StopIteration
            When all channels have been plotted accordingly.
        """
        # if all channels have been plotted, raise StopIteration
        if self.row_idx >= self.axs.shape[0] \
                and self.col_idx >= self.axs.shape[1]:
            raise StopIteration

        # if the current axes is empty, skip it
        if self.row_idx in self.empty_rows :
            self.row_idx += 1
            return self.__next__()

        if self.col_idx in self.empty_cols:
            self.col_idx += 1
            return self.__next__()

        # get the next channel to be plotted
        self.row_idx += 1
        if self.row_idx >= self.axs.shape[0]:
            self.row_idx = 0
            self.col_idx += 1
            return self.__next__()

        ax = self.axs[self.row_idx, self.col_idx]
        row = self.d

        return 

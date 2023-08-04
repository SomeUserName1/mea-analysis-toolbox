import numpy as np

from model.data import Data

class MEAGridPlotIterator:
    """
    An iterator class that iterates through channels to be plotted on a grid.
    """
    def __init__(self, data: Data) -> None:
        """
        Checks which electrodes are to be plotted, if the ground/corner electrodes
        should be plotted too and when rows can be skipped (i.e. when there are multiple 
        adjacent non-selected rows/columns.

        Parameters
        ----------
        data: model.data.Data
           The Data object containing channel data, selected channels
           and metadata to be plotted.
        """
        self.row_offset = 0
        self.col_offset = 0
        self.row_idx = 0
        self.col_idx = -1 
        # See which electrodes are selected and what has to be plotted, such
        # that the grid is as small as possible. Also check if corners are
        # contained, to draw them if neccessary.
        self.grid_sz = int(np.sqrt(data.n_mea_electrodes))
        sel_e = data.selected_electrodes
        print(sel_e)
        names = data.get_sel_names()
        self.crnrs = data.ground_els
        adj_els = [[self.crnrs[0] + 1, self.crnrs[0] + self.grid_sz - 2], # -1 ground.
                   [self.crnrs[1] - 2, self.crnrs[1] + self.grid_sz - 2],
                   [self.crnrs[2] - 2, self.crnrs[2] - self.grid_sz - 2],
                   [self.crnrs[3] - 5, self.crnrs[3] - self.grid_sz - 3]]
        plotted = np.zeros((self.grid_sz, self.grid_sz))
        for i in range(self.grid_sz):
            for j in range(self.grid_sz):
                idx = i * self.grid_sz + j
                if idx >= self.grid_sz * self.grid_sz:
                    break
                # if the channel is a corner and adjacent channels are selected
                # mark it as to be plotted
                if idx == self.crnrs[0] and all([el in sel_e for el in adj_els[0]]) \
                    or idx == self.crnrs[1] and all([el in sel_e for el in adj_els[1]]) \
                    or idx == self.crnrs[2] and all([el in sel_e for el in adj_els[2]]) \
                    or idx == self.crnrs[3] and all([el in sel_e for el in adj_els[3]]):
                        plotted[i, j] = 1
                # If the channel is selected, mark it as to be plotted
                if idx in sel_e:
                    plotted[i, j] = 1

        # check for empty rows and columns.
        # if its the first empty row/col after a non-empty one print it st.
        # the cut in the grid is visible
        # for every following empty row/col omit it from the grid, st. the
        # grid is as small as possible
        self.empty_rows = []
        self.empty_cols = []
        row_sums = np.sum(plotted, axis=1)
        col_sums = np.sum(plotted, axis=0)
        print(plotted)
        print(col_sums)
        for sums, empties in zip([row_sums, col_sums], [self.empty_rows, self.empty_cols]):
            prev = 0
            last = np.nonzero(sums)[-1][0]
            print(last)
            for i, s in enumerate(sums):
                if i == last:
                    empties.extend([i for i in range(last + 1, self.grid_sz)]) 
                    break

                if prev == 0 and s == 0:
                    empties.append(i)

                prev = s

        # adjust the grid size according to the empty rows and cols
        # and create the figure and axes objects
        self.grid_y = self.grid_sz - len(self.empty_rows)
        self.grid_x = self.grid_sz - len(self.empty_cols)

        print(self.empty_rows)
        print(self.empty_cols)


    def __iter__(self):
        """
        Implement the iterator protocol.

        Returns
        -------
        self: GridPlotIterator
        """
        return self


    def __next__(self) -> tuple[int, int]:
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
        # get the next channel to be plotted
        self.col_idx += 1

        print(f"caling next row {self.row_idx}, col {self.col_idx}, grid_x {self.grid_x}, grid_y {self.grid_y} row_o {self.row_offset} col_o {self.col_offset}")

        # if all channels have been plotted, raise StopIteration
        if self.row_idx - self.row_offset >= self.grid_y \
                and self.col_idx - self.col_offset >= self.grid_x:
            raise StopIteration

        # if the current row is empty skip it by setting the col idx to the end
        if self.row_idx in self.empty_rows:
            self.row_offset += 1
            self.col_idx = self.grid_x
        
        # if we are past the last electrode of the grid (or are skipping a row)
        # go to the next row
        if self.col_idx - self.col_offset >= self.grid_x:
            self.row_idx += 1
            self.col_idx = -1
            self.col_offset = 0
            return self.__next__()

        # if the current col is empty, skip it
        if self.col_idx in self.empty_cols:
            self.col_offset += 1
            return self.__next__()

        # if it's a corner/ ground electrode skip it
        if self.row_idx * self.grid_sz + self.col_idx in self.crnrs:
            return self.__next__()

        return self.row_idx - self.row_offset, self.col_idx - self.col_offset

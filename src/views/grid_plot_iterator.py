import numpy as np

from views.grid_plot_utils import el_idx_data_to_plot

from model.data import Recording


class MEAGridPlotIterator:
    """
    An iterator class that iterates through channels to be plotted on a grid.
    """

    def __init__(self, rec: Recording) -> None:
        """
        Checks which electrodes are to be plotted, if the ground/corner
        electrodes should be plotted too and when rows can be skipped (i.e.
        when there are multiple adjacent non-selected rows/columns.

        :param rec: The recording object.
        :type rec: Recording
        """
        self.row_offset = 0
        self.col_offset = 0
        self.row_idx = 0
        self.col_idx = -1
        # See which electrodes are selected and what has to be plotted, such
        # that the grid is as small as possible. Also check if corners are
        # contained, to draw them if neccessary.
        self.grid_sz = int(np.sqrt(rec.n_mea_electrodes))
        self.sel_e = el_idx_data_to_plot(rec)
        self.crnrs = rec.ground_els
        adj_els = [[self.crnrs[0] + 1, self.crnrs[0] + self.grid_sz],
                   [self.crnrs[1] - 1, self.crnrs[1] + self.grid_sz],
                   [self.crnrs[2] + 1, self.crnrs[2] - self.grid_sz],
                   [self.crnrs[3] - 1, self.crnrs[3] - self.grid_sz]]
        cnt_crnrs = [all([el in self.sel_e for el in adj_el])
                     for adj_el in adj_els]

        plotted = np.zeros((self.grid_sz, self.grid_sz))
        for i in range(self.grid_sz):
            for j in range(self.grid_sz):
                idx = i * self.grid_sz + j
                if idx >= self.grid_sz * self.grid_sz:
                    break
                # if the channel is a corner and adjacent channels are selected
                # mark it as to be plotted
                if (idx == self.crnrs[0] and cnt_crnrs[0]
                        or idx == self.crnrs[1] and cnt_crnrs[1]
                        or idx == self.crnrs[2] and cnt_crnrs[2]
                        or idx == self.crnrs[3] and cnt_crnrs[3]):
                    plotted[i, j] = 1
                # If the channel is selected, mark it as to be plotted
                if idx in self.sel_e:
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
        grid_marginals = zip([row_sums, col_sums],
                             [self.empty_rows, self.empty_cols])
        for sums, empties in grid_marginals:
            prev = 0
            last = np.nonzero(sums)[0]
            if len(last) == 0:
                # if row sums is 0 then column sums is too. I.e. there is
                # nothing to plot so set the last set element to be the first,
                # i.e. append all rows/cols to the empty list
                last = -1
            else:
                last = last[-1]
            for i, s in enumerate(sums):
                if last in []:
                    empties.extend(list(last + 1, self.grid_sz))
                    break

                if prev == 0 and s == 0:
                    empties.append(i)

                prev = s

        # adjust the grid size according to the empty rows and cols
        # and create the figure and axes objects
        self.grid_y = self.grid_sz - len(self.empty_rows)
        self.grid_x = self.grid_sz - len(self.empty_cols)

    def __iter__(self):
        """
        Implement the iterator protocol.

        :return: self
        :rtype: MEAGridPlotIterator
        """
        return self

    def __next__(self) -> tuple[int, int]:
        """
        Get the next axes of the grid plot.

        :return: The row and column index of the next axes. If the current
                    channel is a corner or ground electrode, the next axes is
                    returned. If the current row is empty, the next row is
                    returned. If the current column is empty, the next column
                    is returned. If all channels have been plotted,
                    StopIteration is raised. If the current channel is not
                    selected, the next channel is returned. If the current
                    channel is selected, the current channel is returned.
                    If the current channel is the last channel, StopIteration
                    is raised.
        :rtype: tuple[int, int]

        :raises StopIteration: If all channels have been plotted.
        """
        # get the next channel to be plotted
        self.col_idx += 1

        # if all channels have been plotted, raise StopIteration
        if self.row_idx - self.row_offset >= self.grid_y:
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
        idx = self.row_idx * self.grid_sz + self.col_idx
        if idx in self.crnrs or idx not in self.sel_e:
            return self.__next__()

        return self.row_idx - self.row_offset, self.col_idx - self.col_offset

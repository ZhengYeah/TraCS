import numpy as np


def discrete_location_grid(granularity, x_max=1, y_max=1):
    """
    Generate a grid of discrete locations in 2D unit square
    """
    # include the boundary: granularity + 1 points each dimension
    # if x_max = 1, y_max = 1, granularity = 10, then there are 11 interpolation points in x and y-axis
    x = np.linspace(0, x_max, granularity + 1, endpoint=True)
    y = np.linspace(0, y_max, granularity + 1, endpoint=True)
    # round to 4 decimal places
    x = np.round(x, 5)
    y = np.round(y, 5)
    return [(i, j) for i in x for j in y]


class GridLocationSpace:
    def __init__(self, granularity, x_max=1, y_max=1):
        self.granularity = granularity
        self.x_max = x_max
        self.y_max = y_max
        self.location_space = self.form_cells(x_max, y_max, granularity)

    def form_cells(self, x_max, y_max, granularity) -> list:
        """
        Form the cells in the location space based on the granularity and the max x and y values.
        If the granularity is 10 * 10,
        x_max is 1, y_max is 1, then there are 10 * 10 = 100 cells in the location space, each cell is a square with side length 0.1,
        and the representative point of each cell is the center of the cell.
        """
        cells = []
        for i in range(granularity):
            for j in range(granularity):
                cell_x = i * (x_max / granularity) + (x_max / (2 * granularity))
                cell_y = j * (y_max / granularity) + (y_max / (2 * granularity))
                cells.append((np.round(cell_x, 5), np.round(cell_y, 5)))
        return cells

    def get_location_space(self):
        return self.location_space

    def uniform_sample_from_a_cell(self, cell_x, cell_y) -> tuple:
        """
        Sample a location uniformly from the cell defined by its representative point (cell_x, cell_y).
        """
        cell_size_x = self.x_max / self.granularity
        cell_size_y = self.y_max / self.granularity
        x_min = cell_x - (cell_size_x / 2)
        x_max = cell_x + (cell_size_x / 2)
        y_min = cell_y - (cell_size_y / 2)
        y_max = cell_y + (cell_size_y / 2)
        x_sampled = np.random.uniform(x_min, x_max)
        y_sampled = np.random.uniform(y_min, y_max)
        return (x_sampled, y_sampled)

    def location_to_cell(self, location: tuple) -> tuple:
        """
        Map a location to the cell it belongs to. Return the representative point of the cell.
        """
        cell_size_x = self.x_max / self.granularity
        cell_size_y = self.y_max / self.granularity
        cell_x = int(location[0] // cell_size_x) * cell_size_x + (cell_size_x / 2)
        cell_y = int(location[1] // cell_size_y) * cell_size_y + (cell_size_y / 2)
        return (cell_x, cell_y)

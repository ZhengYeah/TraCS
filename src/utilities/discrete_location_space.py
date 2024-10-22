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

import numpy as np


def discrete_location_grid(granularity, x_max=1, y_max=1):
    """
    Generate a grid of discrete locations in 2D unit square
    """
    x = np.linspace(0, x_max, granularity + 1)  # include the boundary, granularity + 1 points each dimension
    y = np.linspace(0, y_max, granularity + 1)
    return [(i, j) for i in x for j in y]

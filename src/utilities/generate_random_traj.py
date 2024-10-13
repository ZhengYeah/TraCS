import numpy as np


def generate_random_traj(n_points, x_max=1, y_max=1):
    """
    Generate a random trajectory with n_points points in 2D unit square
    """
    tmp = np.random.rand(n_points, 2)
    tmp[:, 0] *= x_max
    tmp[:, 1] *= y_max
    return tmp


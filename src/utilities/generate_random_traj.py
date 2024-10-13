import numpy as np


def generate_random_traj(n_points):
    """
    Generate a random trajectory with n_points points in 2D unit square
    """
    return np.random.rand(n_points, 2)

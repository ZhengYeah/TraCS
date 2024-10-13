import numpy as np


def l2_distance(traj1, traj2):
    """
    Compute the L2 distance between two trajectories
    """
    return np.linalg.norm(np.array(traj1) - np.array(traj2))

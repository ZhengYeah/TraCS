import numpy as np


def l2_distance(traj_1, traj_2):
    """
    Compute the L2 distance between two locations in two trajectories
    """
    sum_distance = 0
    for i in range(len(traj_1)):
        sum_distance += np.sqrt((traj_1[i][0] - traj_2[i][0]) ** 2 + (traj_1[i][1] - traj_2[i][1]) ** 2)
    return sum_distance


def averaged_l2_distance(traj_1, traj_2):
    """
    Compute the averaged L2 distance between two locations in two trajectories
    """
    return l2_distance(traj_1, traj_2) / len(traj_1)

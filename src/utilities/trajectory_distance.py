import numpy as np


def l2_distance(traj_1, traj_2):
    """
    Compute the L2 distance between two locations in two trajectories
    """
    assert len(traj_1) == len(traj_2)
    sum_distance = 0
    for i in range(len(traj_1)):
        sum_distance += np.sqrt((traj_1[i][0] - traj_2[i][0]) ** 2 + (traj_1[i][1] - traj_2[i][1]) ** 2)
    return sum_distance


def averaged_l2_distance(traj_1, traj_2):
    """
    Compute the averaged L2 distance between two locations in two trajectories
    """
    assert len(traj_1) == len(traj_2)
    return l2_distance(traj_1, traj_2) / len(traj_1)

def range_query_distance(traj_1, traj_2, range_delta):
    """
    Compute the L2 distance between two trajectories within a specified range.
    """
    assert len(traj_1) == len(traj_2)
    count = 0
    for i in range(len(traj_1)):
        distance = np.sqrt((traj_1[i][0] - traj_2[i][0]) ** 2 + (traj_1[i][1] - traj_2[i][1]) ** 2)
        if distance <= range_delta:
            count += 1
    return count / len(traj_1) if len(traj_1) > 0 else 0.0

def hotpot_count(traj, hotpot_set):
    """
    Compute the preservation distance of a trajectory with respect to a set of hotpots.
    """
    count = 0
    for point in traj:
        if tuple(point) in hotpot_set:
            count += 1
    return count

import numpy as np
from copy import deepcopy
from src.ldp_mechanisms import DiscreteMechanism


def srr_perturb(traj, location_space, epsilon, distance_list: "list"):
    assert len(distance_list) == 2
    traj_copy = deepcopy(traj)
    for i in range(len(traj_copy)):
        # forms 3 groups of locations based on distance
        groups = [[], [], []]
        for loc in location_space:
            if np.sqrt((traj_copy[i][0] - loc[0]) ** 2 + (traj_copy[i][1] - loc[1]) ** 2) <= distance_list[0]:
                groups[0].append(loc)
            elif np.sqrt((traj_copy[i][0] - loc[0]) ** 2 + (traj_copy[i][1] - loc[1]) ** 2) <= distance_list[1]:
                groups[1].append(loc)
            else:
                groups[2].append(loc)

        srr_mechanism = DiscreteMechanism(traj_copy[i], epsilon, len(location_space))
        traj_copy[i] = srr_mechanism.srr_3_groups(groups)
    return traj_copy

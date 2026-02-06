import numpy as np
from src.ldp_mechanisms import DiscreteMechanism

pi = np.pi


def tp_bi_direction(loc_1, loc_2, private_loc, location_space, epsilon):
    """
    TP bi-direction perturbation mechanism (location space reduction)
    :param loc_1: former location
    :param loc_2: latter location
    :param private_loc: the private location to be perturbed
    :param location_space: all possible locations (whole location space)
    :param epsilon: privacy budget for each direction
    :return: perturbed location
    """
    direction_1 = np.arctan2(private_loc[1] - loc_1[1], private_loc[0] - loc_1[0])
    if direction_1 < 0:
        direction_1 += 2 * pi
    direction_2 = np.arctan2(private_loc[1] - loc_2[1], private_loc[0] - loc_2[0])
    if direction_2 < 0:
        direction_2 += 2 * pi
    krr_granularity = 6
    krr_direction_1 = DiscreteMechanism(0, epsilon / 2, krr_granularity).krr()
    krr_direction_2 = DiscreteMechanism(0, epsilon / 2, krr_granularity).krr()
    perturbed_sector_1 = ((direction_1 - pi / krr_granularity + krr_direction_1 * (2 * pi / krr_granularity)) % (2 * pi),
                          (direction_1 + pi / krr_granularity + krr_direction_1 * (2 * pi / krr_granularity)) % (2 * pi))
    perturbed_sector_2 = ((direction_2 - pi / krr_granularity + krr_direction_2 * (2 * pi / krr_granularity)) % (2 * pi),
                          (direction_2 + pi / krr_granularity + krr_direction_2 * (2 * pi / krr_granularity)) % (2 * pi))
    # reduced location space
    reduced_location_space = []
    for loc in location_space:
        candidate_direction_1 = np.arctan2(loc[1] - loc_1[1], loc[0] - loc_1[0])
        if candidate_direction_1 < 0:
            candidate_direction_1 += 2 * pi
        candidate_direction_2 = np.arctan2(loc[1] - loc_2[1], loc[0] - loc_2[0])
        if candidate_direction_2 < 0:
            candidate_direction_2 += 2 * pi
        # check if the candidate direction is in the perturbed sector
        # note: the perturbed sector may cross the 0 degree
        if perturbed_sector_1[1] > perturbed_sector_1[0]:
            if perturbed_sector_2[1] > perturbed_sector_2[0]:
                if perturbed_sector_1[0] <= candidate_direction_1 <= perturbed_sector_1[1] and \
                        perturbed_sector_2[0] <= candidate_direction_2 <= perturbed_sector_2[1]:
                    reduced_location_space.append(loc)
            else:
                if perturbed_sector_1[0] <= candidate_direction_1 <= perturbed_sector_1[1] and \
                        (perturbed_sector_2[0] <= candidate_direction_2 or candidate_direction_2 <= perturbed_sector_2[1]):
                    reduced_location_space.append(loc)
        else:
            if perturbed_sector_2[1] > perturbed_sector_2[0]:
                if (perturbed_sector_1[0] <= candidate_direction_1 or candidate_direction_1 <= perturbed_sector_1[1]) and \
                        perturbed_sector_2[0] <= candidate_direction_2 <= perturbed_sector_2[1]:
                    reduced_location_space.append(loc)
            else:
                if (perturbed_sector_1[0] <= candidate_direction_1 or candidate_direction_1 <= perturbed_sector_1[1]) and \
                        (perturbed_sector_2[0] <= candidate_direction_2 or candidate_direction_2 <= perturbed_sector_2[1]):
                    reduced_location_space.append(loc)
    # perturb the private location
    if len(reduced_location_space) == 0:
        em_mechanism = DiscreteMechanism(private_loc, epsilon / 3, len(location_space))
        return em_mechanism.exp_mechanism_loc(location_space), location_space
    else:
        em_mechanism = DiscreteMechanism(private_loc, epsilon / 3, len(reduced_location_space))
        return em_mechanism.exp_mechanism_loc(reduced_location_space), reduced_location_space


def merge_traj(traj_1, traj_2, location_space):
    length = len(traj_1)
    # # exclude traj_1 and traj_2
    # # logic error: always outputs traj_1 or traj_2.
    # # Because traj_1 and traj_2 can be the same when epsilon is large, it should be able to output the locations in traj_1 and traj_2.
    # location_space = [loc for loc in location_space if loc not in traj_1 and loc not in traj_2]
    merged_traj = []
    for i in range(length):
        # find the nearest location to traj_1[i] and traj_2[i]
        nearest_loc = min(location_space, key=lambda x: np.linalg.norm(np.array(x) - np.array(traj_1[i])) + np.linalg.norm(np.array(x) - np.array(traj_2[i])))
        merged_traj.append(nearest_loc)
    return merged_traj


def tp_perturb(traj, location_space, epsilon):
    """
    TP perturbation mechanism
    :param traj: private trajectory
    :param location_space: all possible locations
    :param epsilon: here epsilon is the privacy budget for each location
    :return: the perturbed trajectory
    ========
    Epsilon: In TP, they use epsilon for a trajectory.
    Each copy of the trajectory takes 0.5 epsilon. In each copy, index locations take 0.125 * 0.5 epsilon,
    direction perturbation takes 0.75 * 0.5 epsilon,
    and the perturbation in the reduced location space takes 0.125 * 0.5 epsilon. (page 6, Algorithm 2 in the paper)
    In our implementation, the above is equivalent to use 0.5 epsilon for each location copy.
    In each location copy, index locations take 0.5 epsilon,
    two direction perturbations belongs to the pivot location and takes 0.5 * 0.75 / 2 epsilon,
    and the remained 0.5 *  0.25 epsilon is used for the perturbation in the reduced location space.
    ========
    Extension to trajectory-level epsilon:
    we can use the same epsilon for each location, which is equivalent to using trajectory-level epsilon = len(traj) * epsilon.
    In this case, the privacy parameter allocation is the same as ATP in the paper.
    """
    length = len(traj)
    traj_copy_1 = traj.copy()
    traj_copy_2 = traj.copy()
    # perturb odd points in traj_copy_1
    for i in range(1, len(traj_copy_1), 2):
        traj_copy_1[i] = DiscreteMechanism(traj_copy_1[i], epsilon * 0.5, length).exp_mechanism_loc(location_space)
    # perturb even points in traj_copy_2
    for i in range(0, len(traj_copy_2), 2):
        traj_copy_2[i] = DiscreteMechanism(traj_copy_2[i], epsilon * 0.5, length).exp_mechanism_loc(location_space)
    # direction perturbation
    for i in range(2, len(traj_copy_1) - 1, 2):
        traj_copy_1[i], _ = tp_bi_direction(traj_copy_1[i-1], traj_copy_1[i+1], traj_copy_1[i], location_space, epsilon * 0.5 * 0.75)
    for i in range(1, len(traj_copy_2) - 1, 2):
        traj_copy_2[i], _ = tp_bi_direction(traj_copy_2[i-1], traj_copy_2[i+1], traj_copy_2[i], location_space, epsilon * 0.5 * 0.75)
    # perturb the first and last points
    traj_copy_1[0] = DiscreteMechanism(traj_copy_1[0], epsilon * 0.5, length).exp_mechanism_loc(location_space)
    if len(traj_copy_1) % 2 == 0:
        traj_copy_1[-1] = DiscreteMechanism(traj_copy_1[-1], epsilon * 0.5, length).exp_mechanism_loc(location_space)
    else:
        traj_copy_2[-1] = DiscreteMechanism(traj_copy_2[-1], epsilon * 0.5, length).exp_mechanism_loc(location_space)
    # merge traj_copy_1 and traj_copy_2
    return merge_traj(traj_copy_1, traj_copy_2, location_space)

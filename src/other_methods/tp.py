import numpy as np
from src.ldp_mechanisms import DiscreteMechanism
from src.utilities.discrete_location_space import discrete_location_grid

pi = np.pi


def tp_bi_direction(loc_1, loc_2, private_loc, location_space, epsilon):
    direction_1 = np.arctan2(private_loc[1] - loc_1[1], private_loc[0] - loc_1[0])
    if direction_1 < 0:
        direction_1 += 2 * pi
    direction_2 = np.arctan2(loc_2[1] - private_loc[1], loc_2[0] - private_loc[0])
    if direction_2 < 0:
        direction_2 += 2 * pi
    krr_granularity = 6
    krr_direction_1 = DiscreteMechanism(0, epsilon, krr_granularity).krr()
    krr_direction_2 = DiscreteMechanism(0, epsilon, krr_granularity).krr()
    perturbed_sector_1 = ((direction_1 - pi / krr_granularity + krr_direction_1 * (2 * pi / krr_granularity)) % (2 * pi),
                          (direction_1 + pi / krr_granularity + krr_direction_1 * (2 * pi / krr_granularity)) % (2 * pi))
    perturbed_sector_2 = ((direction_2 - pi / krr_granularity + krr_direction_2 * (2 * pi / krr_granularity)) % (2 * pi),
                            (direction_2 + pi / krr_granularity + krr_direction_2 * (2 * pi / krr_granularity)) % (2 * pi))
    # reduced location space
    reduced_location_space = []
    for loc in location_space:
        if perturbed_sector_1[0] <= np.arctan2(loc[1] - loc_1[1], loc[0] - loc_1[0]) <= perturbed_sector_1[1] and \
           perturbed_sector_2[0] <= np.arctan2(loc_2[1] - loc[1], loc_2[0] - loc[0]) <= perturbed_sector_2[1]:
            reduced_location_space.append(loc)
    # perturb the private location
    if len(reduced_location_space) == 0:
        em_mechanism = DiscreteMechanism(private_loc, epsilon, len(location_space))
        return em_mechanism.exp_mechanism_loc(location_space)
    else:
        em_mechanism = DiscreteMechanism(private_loc, epsilon, len(reduced_location_space))
        return em_mechanism.exp_mechanism_loc(reduced_location_space)


def merge_traj(traj_1, traj_2, location_space):
    length = len(traj_1)
    merged_traj = []
    for i in range(length):
        # find the nearest location to traj_1[i] and traj_2[i]
        nearest_loc = min(location_space, key=lambda x: np.linalg.norm(x - traj_1[i]) + np.linalg.norm(x - traj_2[i]))
        merged_traj.append(nearest_loc)


def tp_perturb(traj, location_space, epsilon):
    length = len(traj)
    traj_copy_1 = traj.deepcopy()
    traj_copy_2 = traj.deepcopy()
    # perturb odd points in traj_copy_1
    for i in range(1, len(traj_copy_1), 2):
        traj_copy_1[i] = DiscreteMechanism(traj_copy_1[i], epsilon, length).exp_mechanism_loc(location_space)
    # perturb even points in traj_copy_2
    for i in range(0, len(traj_copy_2), 2):
        traj_copy_2[i] = DiscreteMechanism(traj_copy_2[i], epsilon, length).exp_mechanism_loc(location_space)
    # direction perturbation
    for i in range(2, len(traj_copy_1), 2):
        tp_bi_direction(traj_copy_1[i - 1], traj_copy_2[i], traj_copy_1[i], location_space, epsilon)
    for i in range(2, len(traj_copy_2), 2):
        tp_bi_direction(traj_copy_2[i - 1], traj_copy_1[i], traj_copy_2[i], location_space, epsilon)
    # merge traj_copy_1 and traj_copy_2
    return merge_traj(traj_copy_1, traj_copy_2, location_space)


if __name__ == "__main__":
    location_space = discrete_location_grid(10, x_max=1, y_max=1)

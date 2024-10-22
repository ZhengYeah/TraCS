import numpy as np
from copy import deepcopy
from src.ldp_mechanisms import DiscreteMechanism


def reachable_locations(private_loc, location_space, theta):
    """
    Location space reduction
    :param private_loc: the location to be perturbed
    :param location_space: all possible locations (whole location space)
    :param theta: the threshold of reachable distance
    :return: reachable locations
    """
    reachable_loc = []
    for loc in location_space:
        if np.sqrt((private_loc[0] - loc[0]) ** 2 + (private_loc[1] - loc[1]) ** 2) <= theta:
            reachable_loc.append(loc)
    return reachable_loc


def ngram_perturb(traj, location_space, epsilon, theta):
    traj_copy = deepcopy(traj)
    for i in range(len(traj_copy)):
        # location space reduction with threshold theta
        reachable_loc = reachable_locations(traj_copy[i], location_space, theta)
        # perturb the private location
        em_mechanism = DiscreteMechanism(traj_copy[i], epsilon, len(reachable_loc))
        traj_copy[i] = em_mechanism.exp_mechanism_loc(reachable_loc)
    return traj_copy

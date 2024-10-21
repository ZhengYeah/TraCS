import numpy as np
from src.ldp_mechanisms import DiscreteMechanism


def reachable_locations(loc, location_space, epsilon):
    """
    Compute the set of reachable locations from a given location
    """
    reachable_loc = []
    for loc_ in location_space:
        if np.linalg.norm(loc - loc_) <= epsilon:
            reachable_loc.append(loc_)
    return reachable_loc


def ngram_perturb(traj, location_space, epsilon):
    length = len(traj)
    traj_copy = traj.deepcopy()
    for i in range(1, len(traj_copy)):
        reachable_loc = reachable_locations(traj_copy[i - 1], location_space, epsilon)
        if len(reachable_loc) == 0:
            em_mechanism = DiscreteMechanism(traj_copy[i], epsilon, len(location_space))
            traj_copy[i] = em_mechanism.exp_mechanism_loc(location_space)
        else:
            em_mechanism = DiscreteMechanism(traj_copy[i], epsilon, len(reachable_loc))
            traj_copy[i] = em_mechanism.exp_mechanism_loc(reachable_loc)
    return traj_copy

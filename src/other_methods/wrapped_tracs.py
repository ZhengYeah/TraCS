import numpy as np
from src.perturbation_tracs import DirectionDistancePerturbation, CoordinatePerturbation


def tracs_d(traj, epsilon, epsilon_d, x_max=1, y_max=1):
    num_locations = len(traj)
    traj_add_dummy = np.vstack((np.array([0, 0]), traj))
    perturbed_trajectory_tracs_d = traj_add_dummy.copy()
    for i in range(num_locations):
        # tracs-d
        ref_location_1 = (perturbed_trajectory_tracs_d[i][0], perturbed_trajectory_tracs_d[i][1])
        location_1 = (perturbed_trajectory_tracs_d[i + 1][0], perturbed_trajectory_tracs_d[i + 1][1])
        tracs_d = DirectionDistancePerturbation(ref_location_1, location_1, epsilon, epsilon_d, x_max, y_max)
        tracs_d.perturb()
        perturbed_trajectory_tracs_d[i + 1] = tracs_d.perturbed_location
    return perturbed_trajectory_tracs_d[1:]


def tracs_c(traj, epsilon, epsilon_1, x_max=1, y_max=1):
    num_locations = len(traj)
    perturbed_trajectory_tracs_c = traj.copy()
    for i in range(num_locations):
        tracs_c = CoordinatePerturbation((traj[i][0], traj[i][1]), epsilon, epsilon_1, x_max, y_max)
        tracs_c.perturb()
        perturbed_trajectory_tracs_c[i] = tracs_c.perturbed_location
    return perturbed_trajectory_tracs_c

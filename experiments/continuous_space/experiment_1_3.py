import numpy as np
from copy import deepcopy
from src.methods.strawman import strawman_perturbation
from src.utilities.generate_random_traj import generate_random_traj
from src.utilities.trajectory_distance import averaged_l2_distance
from src.perturbation_tracs import DirectionDistancePerturbation, CoordinatePerturbation

pi = np.pi
x_max, y_max = 2, 10

def tracs_d_vs_tracs_c(epsilon, trajectory):
    """
    comparison of TRACS-D and Strawman on a random trajectory with n_points points
    """
    # tracs-c
    num_locations = len(trajectory)
    perturbed_trajectory_tracs_c = deepcopy(trajectory)
    epsilon_1 = epsilon / 2
    for i in range(num_locations):
        tracs_c = CoordinatePerturbation((trajectory[i][0], trajectory[i][1]), epsilon, epsilon_1, x_max, y_max)
        tracs_c.perturb()
        perturbed_trajectory_tracs_c[i] = tracs_c.perturbed_location
    # tracs-d and strawman
    # add a dummy ref location to the head of the trajectory
    trajectory = np.vstack((np.array([0, 0]), trajectory))
    perturbed_trajectory_tracs_d = deepcopy(trajectory)
    perturbed_trajectory_strawman = deepcopy(trajectory)
    for i in range(num_locations):
        # tracs-d
        ref_location_1 = (perturbed_trajectory_tracs_d[i][0], perturbed_trajectory_tracs_d[i][1])
        location_1 = (perturbed_trajectory_tracs_d[i + 1][0], perturbed_trajectory_tracs_d[i + 1][1])
        epsilon_d = pi / (pi + 1) * epsilon
        tracs_d = DirectionDistancePerturbation(ref_location_1, location_1, epsilon, epsilon_d, x_max, y_max)
        tracs_d.perturb()
        perturbed_trajectory_tracs_d[i + 1] = tracs_d.perturbed_location
        # strawman
        ref_location_2 = (perturbed_trajectory_strawman[i][0], perturbed_trajectory_strawman[i][1])
        location_2 = (perturbed_trajectory_strawman[i + 1][0], perturbed_trajectory_strawman[i + 1][1])
        perturbed_trajectory_strawman[i + 1] = strawman_perturbation(ref_location_1, location_2, epsilon, x_max, y_max)
    return averaged_l2_distance(trajectory, perturbed_trajectory_tracs_d), averaged_l2_distance(trajectory, perturbed_trajectory_strawman), averaged_l2_distance(trajectory[1:], perturbed_trajectory_tracs_c)


if __name__ == '__main__':
    trajectory = [(0.2, 1), (0.4, 1), (0.6, 1), (0.8, 1), (1, 1), (1.2, 1), (1.4, 1), (1.6, 1), (1.8, 1)]
    for epsilon in [2, 4, 6, 8, 10]:
        distance_tracs_d, distance_strawman, distance_tracs_c = 0, 0, 0
        for _ in range(1000):
            # average distance
            distance_tracs_d += tracs_d_vs_tracs_c(epsilon, trajectory)[0]
            distance_strawman += tracs_d_vs_tracs_c(epsilon, trajectory)[1]
            distance_tracs_c +=tracs_d_vs_tracs_c(epsilon, trajectory)[2]
        print(f"epsilon: {epsilon}, tracs-d: {distance_tracs_d / 1000}, strawman: {distance_strawman / 1000}, tracs-c: {distance_tracs_c / 1000}")
        # save to csv
        with open(f"./results/experiment_1_3.csv", "a") as f:
            # header
            if epsilon == 2:
                # # clear the file
                # f.seek(0)
                # f.truncate()
                f.write("epsilon,tracs_d,strawman,tracs_c\n")
            f.write(f"{epsilon},{distance_tracs_d / 1000},{distance_strawman / 1000},{distance_tracs_c / 1000}\n")
import numpy as np
from copy import deepcopy
from src.methods.strawman import strawman_perturbation
from src.methods.two_d_laplace import ldp_2d_laplace_batch
from src.utilities.generate_random_traj import generate_random_traj
from src.utilities.trajectory_distance import averaged_l2_distance
from src.perturbation_tracs import DirectionDistancePerturbation, CoordinatePerturbation

pi = np.pi


def tracs_vs_strawman(epsilon, num_locations):
    """
    comparison of TRACS-D and Strawman on a random trajectory with n_points points
    """
    trajectory = generate_random_traj(num_locations)
    # tracs-c
    perturbed_trajectory_tracs_c = deepcopy(trajectory)
    perturbed_trajectory_2d_laplace = ldp_2d_laplace_batch(trajectory, epsilon)
    epsilon_1 = epsilon / 2
    for i in range(num_locations):
        tracs_c = CoordinatePerturbation((trajectory[i][0], trajectory[i][1]), epsilon, epsilon_1)
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
        tracs_d = DirectionDistancePerturbation(ref_location_1, location_1, epsilon, epsilon_d)
        tracs_d.perturb()
        perturbed_trajectory_tracs_d[i + 1] = tracs_d.perturbed_location
        # strawman
        ref_location_2 = (perturbed_trajectory_strawman[i][0], perturbed_trajectory_strawman[i][1])
        location_2 = (perturbed_trajectory_strawman[i + 1][0], perturbed_trajectory_strawman[i + 1][1])
        perturbed_trajectory_strawman[i + 1] = strawman_perturbation(ref_location_1, location_2, epsilon)
    return (averaged_l2_distance(trajectory, perturbed_trajectory_tracs_d),
            averaged_l2_distance(trajectory, perturbed_trajectory_strawman),
            averaged_l2_distance(trajectory[1:], perturbed_trajectory_tracs_c), # Remove the dummy head
            averaged_l2_distance(trajectory[1:], perturbed_trajectory_2d_laplace))


if __name__ == '__main__':
    for epsilon in [2, 4, 6, 8, 10]:
        distance_tracs_d, distance_strawman, distance_tracs_c, distance_2d_laplace = 0, 0, 0, 0
        for _ in range(1000):
            # average distance
            distance_tracs_d += tracs_vs_strawman(epsilon, 100)[0]
            distance_strawman += tracs_vs_strawman(epsilon, 100)[1]
            distance_tracs_c += tracs_vs_strawman(epsilon, 100)[2]
            distance_2d_laplace += tracs_vs_strawman(epsilon, 100)[3]
        print(f"epsilon: {epsilon}, tracs-d: {distance_tracs_d / 1000}, strawman: {distance_strawman / 1000}, tracs-c: {distance_tracs_c / 1000}, 2d-laplace: {distance_2d_laplace / 1000}")
        # save to csv
        with open(f"./results/experiment_1_1_add_laplace.csv", "a") as f:
            # header
            if epsilon == 2:
                # # clear the file
                # f.seek(0)
                # f.truncate()
                f.write("epsilon,tracs_d,strawman,tracs_c,2d_laplace\n")
            f.write(f"{epsilon},{distance_tracs_d / 1000},{distance_strawman / 1000},{distance_tracs_c / 1000},{distance_2d_laplace / 1000}\n")

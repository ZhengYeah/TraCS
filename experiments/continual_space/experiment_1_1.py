import numpy as np
from src.other_methods.strawman import strawman_perturbation
from src.utilities.generate_random_traj import generate_random_traj
from src.utilities.trajectory_distance import l2_distance
from src.perturbation_tracs import DirectionDistancePerturbation, CoordinatePerturbation

pi = np.pi


def tracs_d_vs_strawman(epsilon, num_locations):
    """
    comparison of TRACS-D and Strawman on a random trajectory with n_points points
    """
    trajectory = generate_random_traj(num_locations)
    perturbed_trajectory_tracs_d = list(trajectory)
    perturbed_trajectory_strawman = list(trajectory)
    perturbed_trajectory_tracs_c = list(trajectory)
    # add a dummy ref location to the head of the trajectory
    trajectory = np.vstack((np.array([0, 0]), trajectory))
    for i in range(num_locations):
        ref_location = (trajectory[i][0], trajectory[i][1])
        location = (trajectory[i + 1][0], trajectory[i + 1][1])
        # tracs-d
        epsilon_d = pi / (pi + 1) * epsilon
        tracs_d = DirectionDistancePerturbation(ref_location, location, epsilon, epsilon_d)
        tracs_d.perturb()
        perturbed_trajectory_tracs_d[i] = tracs_d.perturbed_location
        # strawman
        perturbed_trajectory_strawman[i] = strawman_perturbation(ref_location, location, epsilon)
        # tracs-c
        epsilon_1 = epsilon / 2
        tracs_c = CoordinatePerturbation(location, epsilon, epsilon_1)
        tracs_c.perturb()
        perturbed_trajectory_tracs_c[i] = tracs_c.perturbed_location
    return l2_distance(trajectory[1:], perturbed_trajectory_tracs_d), l2_distance(trajectory[1:], perturbed_trajectory_strawman), l2_distance(trajectory[1:], perturbed_trajectory_tracs_c)


if __name__ == '__main__':
    for epsilon in [2, 4, 6, 8, 10]:
        distance_tracs_d, distance_strawman, distance_tracs_c = 0, 0, 0
        for _ in range(1000):
            # average distance
            distance_tracs_d += tracs_d_vs_strawman(epsilon, 100)[0]
            distance_strawman += tracs_d_vs_strawman(epsilon, 100)[1]
            distance_tracs_c += tracs_d_vs_strawman(epsilon, 100)[2]
        print(f"epsilon: {epsilon}, tracs-d: {distance_tracs_d / 1000}, strawman: {distance_strawman / 1000}, tracs-c: {distance_tracs_c / 1000}")
        # save to csv
        with open(f"./results/experiment_1_1.csv", "a") as f:
            # header
            if epsilon == 2:
                # # clear the file
                # f.seek(0)
                # f.truncate()
                f.write("epsilon,tracs_d,strawman,tracs_c\n")
            f.write(f"{epsilon},{distance_tracs_d / 1000},{distance_strawman / 1000},{distance_tracs_c / 1000}\n")

import numpy as np
from copy import deepcopy
from src.utilities.generate_random_traj import generate_random_traj
from src.perturbation_tracs import DirectionDistancePerturbation
from src.ldp_mechanisms import DiscreteMechanism

pi = np.pi
x_max, y_max = 1, 1


def direction_tracs_vs_strawman(epsilon, num_locations):
    """
    comparison of TRACS-D and Strawman on a random trajectory with n_points points
    """
    trajectory = generate_random_traj(num_locations, x_max, y_max)
    # tracs-d and strawman
    # add a dummy ref location to the head of the trajectory
    trajectory = np.vstack((np.array([0, 0]), trajectory))
    perturbed_trajectory_tracs_d = deepcopy(trajectory)
    error_tracs_d, error_strawman_6rr, error_strawman_3rr, error_strawman_12rr = 0, 0, 0, 0
    for i in range(num_locations):
        # tracs-d
        ref_location_1 = (perturbed_trajectory_tracs_d[i][0], perturbed_trajectory_tracs_d[i][1])
        location_1 = (perturbed_trajectory_tracs_d[i + 1][0], perturbed_trajectory_tracs_d[i + 1][1])
        epsilon_d = pi / (pi + 1) * epsilon
        tracs_d = DirectionDistancePerturbation(ref_location_1, location_1, epsilon, epsilon_d, x_max, y_max)
        tracs_d._direction_perturbation()
        error_tracs_d += min(abs(tracs_d.perturbed_direction - tracs_d.private_direction), abs(2 * pi - tracs_d.perturbed_direction - tracs_d.private_direction))
        # strawman
        # 3-RR
        private_sector = tracs_d.private_direction // (2 * pi / 3)
        assert 0 <= private_sector <= 2
        mechanism = DiscreteMechanism(private_sector, epsilon_d, 3)
        perturbed_sector = mechanism.krr()
        uniform_sample = np.random.uniform(0, 1)
        perturbed_direction = perturbed_sector * (2 * pi / 3) + uniform_sample * (2 * pi / 3)
        error_strawman_3rr += min(abs(perturbed_direction - tracs_d.private_direction),
                                  abs(2 * pi - perturbed_direction - tracs_d.private_direction))
        # 6-RR
        private_sector = tracs_d.private_direction // (pi / 3)
        assert 0 <= private_sector <= 5
        mechanism = DiscreteMechanism(private_sector, epsilon_d, 6)
        perturbed_sector = mechanism.krr()
        uniform_sample = np.random.uniform(0, 1)
        perturbed_direction = perturbed_sector * (pi / 3) + uniform_sample * (pi / 3)
        error_strawman_6rr += min(abs(perturbed_direction - tracs_d.private_direction),
                                  abs(2 * pi - perturbed_direction - tracs_d.private_direction))

        # 12-RR
        private_sector = tracs_d.private_direction // (pi / 6)
        assert 0 <= private_sector <= 11
        mechanism = DiscreteMechanism(private_sector, epsilon_d, 12)
        perturbed_sector = mechanism.krr()
        uniform_sample = np.random.uniform(0, 1)
        perturbed_direction = perturbed_sector * (pi / 6) + uniform_sample * (pi / 6)
        error_strawman_12rr += min(abs(perturbed_direction - tracs_d.private_direction),
                                   abs(2 * pi - perturbed_direction - tracs_d.private_direction))
    return error_tracs_d / num_locations, error_strawman_3rr / num_locations, error_strawman_6rr / num_locations, error_strawman_12rr / num_locations


if __name__ == '__main__':
    for epsilon in [2, 4, 6, 8, 10]:
        error_tracs_d, error_strawman_6rr, error_strawman_3rr, error_strawman_12rr = 0, 0, 0, 0
        for _ in range(1000):
            # average error
            error_tracs_d += direction_tracs_vs_strawman(epsilon, 100)[0]
            error_strawman_3rr += direction_tracs_vs_strawman(epsilon, 100)[1]
            error_strawman_6rr += direction_tracs_vs_strawman(epsilon, 100)[2]
            error_strawman_12rr += direction_tracs_vs_strawman(epsilon, 100)[3]
        print(f"epsilon: {epsilon}, tracs-d: {error_tracs_d / 1000}, strawman-3rr: {error_strawman_3rr / 1000}, strawman-6rr: {error_strawman_6rr / 1000}, strawman-12rr: {error_strawman_12rr / 1000}")
        # save to csv
        with open(f"./results/experiment_2.csv", "a") as f:
            # header
            if epsilon == 2:
                # # clear the file
                # f.seek(0)
                # f.truncate()
                f.write("epsilon,tracs_d, 3-RR, 6-RR, 12-RR\n")
            f.write(f"{epsilon},{error_tracs_d / 1000},{error_strawman_3rr / 1000},{error_strawman_6rr / 1000},{error_strawman_12rr / 1000}\n")

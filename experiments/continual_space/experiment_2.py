import numpy as np
from copy import deepcopy
from src.utilities.generate_random_traj import generate_random_traj
from src.perturbation_tracs import DirectionDistancePerturbation
from src.ldp_mechanisms import PiecewiseMechanism, DiscreteMechanism

pi = np.pi
x_max, y_max = 1, 1


def direction_tracs_vs_strawman(epsilon):
    """
    comparison of TRACS-D and Strawman on a random trajectory with n_points points
    """
    private_direction = np.random.uniform(0, 2 * pi)
    tracs_direction = PiecewiseMechanism(private_direction, epsilon).circular_perturbation()
    error_tracs_d = min(abs(tracs_direction - private_direction), abs(2 * pi - tracs_direction - private_direction))
    # strawman
    # 3-RR
    private_sector = private_direction // (2 * pi / 3)
    assert 0 <= private_sector <= 2
    direction_3rr = DiscreteMechanism(private_sector, epsilon, 3).krr()
    perturbed_direction = direction_3rr * (2 * pi / 3) + np.random.uniform(0, 1) * (2 * pi / 3)
    error_strawman_3rr = min(abs(perturbed_direction - private_direction), abs(2 * pi - perturbed_direction - private_direction))
    # 6-RR
    private_sector = private_direction // (pi / 3)
    assert 0 <= private_sector <= 5
    direction_6rr = DiscreteMechanism(private_sector, epsilon, 6).krr()
    perturbed_direction = direction_6rr * (pi / 3) + np.random.uniform(0, 1) * (pi / 3)
    error_strawman_6rr = min(abs(perturbed_direction - private_direction), abs(2 * pi - perturbed_direction - private_direction))
    # 12-RR
    private_sector = private_direction // (pi / 6)
    assert 0 <= private_sector <= 11
    direction_12rr = DiscreteMechanism(private_sector, epsilon, 12).krr()
    perturbed_direction = direction_12rr * (pi / 6) + np.random.uniform(0, 1) * (pi / 6)
    error_strawman_12rr = min(abs(perturbed_direction - private_direction), abs(2 * pi - perturbed_direction - private_direction))
    return error_tracs_d, error_strawman_3rr, error_strawman_6rr, error_strawman_12rr





if __name__ == '__main__':
    for epsilon in [2, 4, 6, 8, 10]:
        error_tracs_d, error_strawman_6rr, error_strawman_3rr, error_strawman_12rr = 0, 0, 0, 0
        for _ in range(100):
            # average error
            error_tracs_d += direction_tracs_vs_strawman(epsilon)[0]
            error_strawman_3rr += direction_tracs_vs_strawman(epsilon)[1]
            error_strawman_6rr += direction_tracs_vs_strawman(epsilon)[2]
            error_strawman_12rr += direction_tracs_vs_strawman(epsilon)[3]
        print(f"epsilon: {epsilon}, tracs-d: {error_tracs_d / 100}, 3-RR: {error_strawman_3rr / 100}, 6-RR: {error_strawman_6rr / 100}, 12-RR: {error_strawman_12rr / 100}")
        # # save to csv
        # with open(f"./results/experiment_2.csv", "a") as f:
        #     # header
        #     if epsilon == 2:
        #         # # clear the file
        #         # f.seek(0)
        #         # f.truncate()
        #         f.write("epsilon,tracs_d, 3-RR, 6-RR, 12-RR\n")
        #     f.write(f"{epsilon},{error_tracs_d / 1000},{error_strawman_3rr / 1000},{error_strawman_6rr / 1000},{error_strawman_12rr / 1000}\n")

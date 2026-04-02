import numpy as np
from copy import deepcopy
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.methods.strawman import strawman_perturbation
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
    rows = []
    for epsilon in [2, 4, 6, 8, 10]:
        distance_tracs_d, distance_strawman, distance_tracs_c = 0, 0, 0
        for _ in range(1000):
            # average distance
            distance_tracs_d += tracs_d_vs_tracs_c(epsilon, trajectory)[0]
            distance_strawman += tracs_d_vs_tracs_c(epsilon, trajectory)[1]
            distance_tracs_c += tracs_d_vs_tracs_c(epsilon, trajectory)[2]
        row = {
            "epsilon": epsilon,
            "tracs_d": distance_tracs_d / 1000,
            "strawman": distance_strawman / 1000,
            "tracs_c": distance_tracs_c / 1000,
        }
        rows.append(row)
        print(
            f"epsilon: {epsilon}, tracs-d: {row['tracs_d']}, strawman: {row['strawman']}, tracs-c: {row['tracs_c']}"
        )
    import matplotlib.pyplot as plt
    plt.rcParams["font.size"] = 20
    plt.figure(figsize=(5,5))
    plt.ylim(0, 3.5)
    plt.xticks([2, 4, 6, 8, 10])
    plt.plot([2, 4, 6, 8, 10], [row["tracs_d"] for row in rows], label="TraCS-D", linestyle="-", color="red", marker="o")
    plt.plot([2, 4, 6, 8, 10], [row["tracs_c"] for row in rows], label="TraCS-C", linestyle="-", color="blue", marker="s")
    plt.xlabel(r'Privacy parameter $\varepsilon$')
    plt.ylabel(r"Average error (AE)")
    plt.legend()
    plt.title("Figure 5")
    plt.show()

import numpy as np
from copy import deepcopy
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

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


# simulate_fn: a functional argument
def run_experiment(simulate_fn, epsilons, num_locations, num_runs):
    rows = []
    for epsilon in epsilons:
        distance_tracs_d = 0.0
        distance_strawman = 0.0
        distance_tracs_c = 0.0
        distance_2d_laplace = 0.0
        for _ in range(num_runs):
            tracs_d, strawman, tracs_c, laplace = simulate_fn(epsilon, num_locations)
            distance_tracs_d += tracs_d
            distance_strawman += strawman
            distance_tracs_c += tracs_c
            distance_2d_laplace += laplace

        row = {
            "epsilon": epsilon,
            "tracs_d": distance_tracs_d / num_runs,
            "strawman": distance_strawman / num_runs,
            "tracs_c": distance_tracs_c / num_runs,
            "2d_laplace": distance_2d_laplace / num_runs,
        }
        rows.append(row)
        print(
            f"epsilon: {epsilon}, tracs-d: {row['tracs_d']}, strawman: {row['strawman']}, "
            f"tracs-c: {row['tracs_c']}, 2d-laplace: {row['2d_laplace']}"
        )
    return rows

# plot the results using matplotlib
epsilon = [2, 4, 6, 8, 10]
rows = run_experiment(tracs_vs_strawman, epsilon, 100, 1000)
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 20
plt.figure(figsize=(5,5))
plt.ylim(0, 0.5)
plt.xticks(epsilon)
plt.plot(epsilon, [row["2d_laplace"] for row in rows], label="T-Laplace", linestyle="--", color="gray", marker="^")
plt.plot(epsilon, [row["strawman"] for row in rows], label="Strawman", linestyle="--", color="black", marker="x")
plt.plot(epsilon, [row["tracs_d"] for row in rows], label="TraCS-D", linestyle="-.", color="red", marker="o")
plt.plot(epsilon, [row["tracs_c"] for row in rows], label="TraCS-C", linestyle=":", color="blue", marker="s")
plt.xlabel(r'Privacy parameter $\varepsilon$')
plt.ylabel(r"Average error")
plt.legend()
plt.title("Figure 4a")
print("========")

########
# Figure 4b
########

x_max, y_max = 2.0, 10.0

def tracs_vs_strawman(epsilon, num_locations):
    """
    comparison of TRACS-D and Strawman on a random trajectory with n_points points
    """
    trajectory = generate_random_traj(num_locations, x_max, y_max)
    # tracs-c
    perturbed_trajectory_tracs_c = deepcopy(trajectory)
    # Map [0,x_max]x[0,y_max] to [0,1]x[0,1]
    trajectory_01 = np.array([(loc[0] / x_max, loc[1] / y_max) for loc in trajectory])
    perturbed_trajectory_2d_laplace = ldp_2d_laplace_batch(trajectory_01, epsilon)
    # Map [0,1]x[0,1] to [0,x_max]x[0,y_max]
    # ldp_2d_laplace_batch: List[Tuple[float, float]]
    perturbed_trajectory_2d_laplace = np.array([(loc[0] * x_max, loc[1] * y_max) for loc in perturbed_trajectory_2d_laplace])
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
    return (averaged_l2_distance(trajectory, perturbed_trajectory_tracs_d),
            averaged_l2_distance(trajectory, perturbed_trajectory_strawman),
            averaged_l2_distance(trajectory[1:], perturbed_trajectory_tracs_c), # Remove the dummy head
            averaged_l2_distance(trajectory[1:], perturbed_trajectory_2d_laplace))

rows = run_experiment(tracs_vs_strawman, epsilon, 100, 1000)
plt.figure(figsize=(5,5))
plt.ylim(0, 3.5)
plt.yticks([0.5, 1, 1.5, 2, 2.5, 3])
plt.xticks(epsilon)
plt.plot(epsilon, [row["2d_laplace"] for row in rows], label="T-Laplace", linestyle="--", color="gray", marker="^")
plt.plot(epsilon, [row["strawman"] for row in rows], label="Strawman", linestyle="--", color="black", marker="x")
plt.plot(epsilon, [row["tracs_d"] for row in rows], label="TraCS-D", linestyle="-.", color="red", marker="o")
plt.plot(epsilon, [row["tracs_c"] for row in rows], label="TraCS-C", linestyle=":", color="blue", marker="s")
plt.xlabel(r'Privacy parameter $\varepsilon$')
plt.ylabel(r"Average error")
plt.legend()
plt.title("Figure 4b")
plt.show()
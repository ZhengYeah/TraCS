import numpy as np

from src.perturbation_tracs import DirectionDistancePerturbation, CoordinatePerturbation
from src.methods.wrapped_tracs import tracs_d
from src.utilities.generate_random_traj import generate_random_traj
from src.utilities.trajectory_distance import averaged_l2_distance
from copy import deepcopy

pi = np.pi

def compare_epsilon_d(epsilon_d, epsilon):
    """
    Compare the performance of TRACS-D with different epsilon_d on a random trajectory with 100 points
    """
    trajectory = generate_random_traj(100)
    # add a dummy ref location to the head of the trajectory
    trajectory = np.vstack((np.array([0.01, 0.01]), trajectory))
    perturbed_trajectory_tracs_d = deepcopy(trajectory)
    for i in range(100):
        ref_location = (perturbed_trajectory_tracs_d[i][0], perturbed_trajectory_tracs_d[i][1])
        location = (perturbed_trajectory_tracs_d[i + 1][0], perturbed_trajectory_tracs_d[i + 1][1])
        tracs_d = DirectionDistancePerturbation(ref_location, location, epsilon, epsilon_d)
        tracs_d.perturb()
        perturbed_trajectory_tracs_d[i + 1] = tracs_d.perturbed_location
    return averaged_l2_distance(trajectory, perturbed_trajectory_tracs_d)


if __name__ == '__main__':
    epsilon_list = [5, 8]
    points_num = 30
    results = []
    for epsilon in epsilon_list:
        epsilon_d = np.linspace(1, epsilon - 0.2, points_num, endpoint=True)
        for epsilon_d in epsilon_d:
            distance_tracs_d = 0
            for _ in range(100):
                distance_tracs_d += compare_epsilon_d(epsilon_d, epsilon)
            print(f"epsilon: {epsilon}, epsilon_d percentage: {epsilon_d / epsilon}, tracs-d: {distance_tracs_d / 100}")
            results.append((epsilon, epsilon_d / epsilon, distance_tracs_d / 100))
            # draw the plot
    import matplotlib.pyplot as plt
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 20
    plt.rcParams["figure.figsize"] = (5, 5)
    plt.xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    for epsilon in epsilon_list:
        epsilon_d_percentage = [result[1] for result in results if result[0] == epsilon]
        distance_tracs_d = [result[2] for result in results if result[0] == epsilon]
        # draw the plot, different epsilon with different color and marker
        marker = "x" if epsilon == 8 else "o"
        color = "black" if epsilon == 8 else "blue"
        plt.plot(epsilon_d_percentage, distance_tracs_d, label=r"$\varepsilon = {}$".format(epsilon), linestyle="--", color=color, marker=marker)
    plt.xlabel(r'Percentage of $\varepsilon_d$ allocated to direction perturbation')
    plt.ylabel(r"Average error")
    plt.legend()
    plt.savefig("./results/experiment_epsilon_d.pdf", bbox_inches='tight')
    plt.show()


"""
TraCS-D for circular area
"""

import numpy as np
from src.ldp_mechanisms import PiecewiseMechanism


pi = np.pi

def tracs_d_circular(location, epsilon):
    """
    TraCS-D for circular area [0,1) x [0, 2*pi)
    :param location: represented by (r, theta)
    """
    assert 0 <= location[0] <= 1 and 0 <= location[1] < 2 * pi
    # perturbation
    epsilon_d = pi / (pi + 1) * epsilon
    perturbed_angle = PiecewiseMechanism(location[1], epsilon_d).circular_perturbation()
    perturbed_radius = PiecewiseMechanism(location[0], epsilon - epsilon_d).linear_perturbation()
    return perturbed_radius, perturbed_angle


if __name__ == "__main__":
    location = (0.5, pi)
    epsilon = 5
    set_perturbed_location = set()
    for _ in range(50):
        set_perturbed_location.add(tracs_d_circular(location, epsilon))
    # plot the perturbed locations
    import matplotlib.pyplot as plt
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 20
    plt.rcParams["figure.figsize"] = (5, 5)
    # draw the plot in polar coordinates
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    for perturbed_location in set_perturbed_location:
        ax.plot(perturbed_location[1], perturbed_location[0], 'o')
    ax.set_rticks([0.5, 1])
    theta_labels = ["0", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$", r"$\frac{5\pi}{4}$", r"$\frac{3\pi}{2}$", r"$\frac{7\pi}{4}$"]
    ax.set_xticks([0, pi / 4, pi / 2, 3 * pi / 4, pi, 5 * pi / 4, 3 * pi / 2, 7 * pi / 4])
    ax.set_xticklabels(theta_labels)

    plt.savefig("./results/experiment_4.eps")
    plt.show()

import pytest
import numpy as np
from src.perturbation_tracs import CoordinatePerturbation

pi = np.pi


@pytest.mark.parametrize("location, epsilon", [((1, 2.9), 6),])
def test_coordinate_perturbation(location, epsilon):
    set_perturbed_location = set()
    epsilon_1 = epsilon / 2
    for _ in range(1000):
        perturbation = CoordinatePerturbation(location, epsilon, epsilon_1, 2, 3)
        perturbation.perturb()
        set_perturbed_location.add(perturbation.perturbed_location)
    import matplotlib.pyplot as plt
    plot = plt.figure()
    ax = plot.add_subplot(111)
    x, y = zip(*set_perturbed_location)
    ax.plot(*location, 'ro')
    ax.scatter(x, y)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 3)
    plt.show()

import pytest
import numpy as np
from src.perturbation_tracs import CoordinatePerturbation

pi = np.pi


@pytest.mark.parametrize("location, epsilon", [((0.5, 0.5), 2),])
def test_coordinate_perturbation(location, epsilon):
    set_perturbed_location = set()
    for _ in range(1000):
        perturbation = CoordinatePerturbation(location, epsilon)
        perturbation.perturb()
        set_perturbed_location.add(perturbation.perturbed_location)
    import matplotlib.pyplot as plt
    plot = plt.figure()
    ax = plot.add_subplot(111)
    x, y = zip(*set_perturbed_location)
    ax.plot(*location, 'ro')
    ax.scatter(x, y)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.show()

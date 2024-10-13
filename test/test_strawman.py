import numpy as np
import pytest
from src.other_methods.strawman import strawman_perturbation

pi = np.pi


@pytest.mark.parametrize("ref_location, location, epsilon", [((0.5, 0.5), (0.6, 0.6), 6),])
def test_strawman(ref_location, location, epsilon):
    set_perturbed_location = set()
    for _ in range(1000):
        set_perturbed_location.add(strawman_perturbation(ref_location, location, epsilon))
    import matplotlib.pyplot as plt
    plt.scatter(*zip(*set_perturbed_location))  # pack the tuples into zip
    plt.show()

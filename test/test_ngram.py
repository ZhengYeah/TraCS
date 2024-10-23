import pytest
import numpy as np
from src.utilities.discrete_location_space import discrete_location_grid
from src.methods.ngram import ngram_perturb


pi = np.pi


@pytest.mark.parametrize("traj, location_space, epsilon", [([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)], discrete_location_grid(40, x_max=4, y_max=4), 4)])
def test_ngram_perturb(traj, location_space, epsilon):
    perturbed_traj = ngram_perturb(traj, location_space, epsilon)
    import matplotlib.pyplot as plt
    plt.scatter(*zip(*location_space), c='r')
    plt.plot(*zip(*traj), c='b')
    plt.plot(*zip(*perturbed_traj), c='y')
    plt.show()


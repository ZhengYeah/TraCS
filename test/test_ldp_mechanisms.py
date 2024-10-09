import pytest
import numpy as np
from src.ldp_mechanisms import PiecewiseMechanism, DiscreteMechanism

pi = np.pi


def test_circular_perturbation(self):
    private_val = 0
    epsilon = 1
    perturbed = PiecewiseMechanism(private_val, epsilon).circular_perturbation()
    assert 0 <= perturbed <= 2 * pi




import pytest
import numpy as np
from src.ldp_mechanisms import PiecewiseMechanism, DiscreteMechanism

pi = np.pi
exp = np.e


@pytest.mark.parametrize("private_val, epsilon", [(0, 1), (pi, 1), (2 * pi, 1), (0, 2), (pi, 2), (2 * pi, 2)])
def test_circular_perturbation_func(private_val, epsilon):
    # verify the [l,r] interval
    ref_l = (private_val - pi * (exp ** (epsilon / 2) - 1) / (exp ** epsilon - 1)) % (2 * pi)
    ref_r = (private_val + pi * (exp ** (epsilon / 2) - 1) / (exp ** epsilon - 1)) % (2 * pi)
    ref_p_epsilon = 1 / (2 * pi) * (exp ** (epsilon / 2)) * (2 * pi * (exp ** (epsilon / 2) - 1) / (exp ** epsilon - 1))
    counter = 0
    for i in range(2000):
        perturbed = PiecewiseMechanism(private_val, epsilon).circular_perturbation()
        assert 0 <= perturbed <= 2 * pi
        if ref_l > ref_r:
            if ref_l <= perturbed <= 2 * pi or 0 <= perturbed <= ref_r:
                counter += 1
        else:
            if ref_l <= perturbed <= ref_r:
                counter += 1
    # print()
    # print(f"statistical frequency of [{ref_l}, {ref_r}]: {counter / 2000}; expected: {ref_p_epsilon}")
    assert abs(counter / 2000 - ref_p_epsilon) < 0.05


@pytest.mark.parametrize("private_val, epsilon", [(0, 1), (1, 1), (0, 2), (1, 2), (0.2, 1), (0.5, 1)])
def test_linear_perturbation_func(private_val, epsilon):
    # verify the [l,r] interval
    C = (exp ** (epsilon / 2) - 1) / (2 * exp ** epsilon - 2)
    p = exp ** (epsilon / 2)
    ref_p_epsilon = 2 * C * p
    counter = 0
    for i in range(2000):
        perturbed = PiecewiseMechanism(private_val, epsilon).linear_perturbation()
        assert 0 <= perturbed <= 1
        if private_val < C:
            if 0 <= perturbed <= 2 * C:
                counter += 1
        elif private_val > 1 - C:
            if 1 - 2 * C <= perturbed <= 1:
                counter += 1
        else:
            if private_val - C <= perturbed <= private_val + C:
                counter += 1
    # print()
    # print(f"statistical frequency of [{private_val - C}, {private_val + C}]: {counter / 2000}; expected: {ref_p_epsilon}")
    assert abs(counter / 2000 - ref_p_epsilon) < 0.05




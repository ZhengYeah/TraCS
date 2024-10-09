import pytest
import numpy as np
from src.ldp_mechanisms import PiecewiseMechanism, DiscreteMechanism

pi = np.pi
exp = np.e


def test_circular_perturbation():
    private_val = 0
    epsilon = 1
    # verify the [l,r] interval
    ref_l = private_val - pi * (exp ** (epsilon / 2) - 1) / (exp ** epsilon - 1) % (2 * pi)
    ref_r = private_val + pi * (exp ** (epsilon / 2) - 1) / (exp ** epsilon - 1) % (2 * pi)
    ref_p_epsilon = 1 / (2 * pi) * (exp ** (epsilon / 2)) * (ref_r - ref_l)
    counter = 0
    for i in range(1000):
        perturbed = PiecewiseMechanism(private_val, epsilon).circular_perturbation()
        assert 0 <= perturbed <= 2 * pi
        if perturbed >= ref_l or perturbed <= ref_r:
            counter += 1
    print(f"statistical frequency of [ref_l, ref_r]: {counter / 1000}; expected: {ref_p_epsilon}")


def test_linear_perturbation():
    private_val = 0
    epsilon = 1
    C = (exp ** (epsilon / 2) - 1) / (2 * exp ** epsilon - 2)
    p = exp ** (epsilon / 2)
    counter = 0
    for i in range(1000):
        perturbed = PiecewiseMechanism(private_val, epsilon).linear_perturbation()
        assert 0 <= perturbed <= 1
        if 0 <= perturbed <= C or 1 - C <= perturbed <= 1:
            counter += 1
    print(f"statistical frequency of [0, C] or [1 - C, 1]: {counter / 1000}; expected: {2 * C * p}")




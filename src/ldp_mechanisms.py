import numpy as np
import random

pi = np.pi
exp = np.e


class PiecewiseMechanism:
    def __init__(self, private_val, epsilon):
        self.private_val = private_val
        self.epsilon = epsilon

    def circular_perturbation(self) -> "float":
        """
        Direction perturbation defined in the paper (Definition 3.1)
        :return: the perturbed direction
        """
        assert 0 <= self.private_val <= 2 * pi + 1e-6
        # insight: sample from \mechanism(\pi, \epsilon) and then add self.private_val to the sample
        l = pi * (1 - (exp ** (self.epsilon / 2) - 1) / (exp ** self.epsilon - 1))
        r = pi * (1 + (exp ** (self.epsilon / 2) - 1) / (exp ** self.epsilon - 1))
        assert 0 <= l < r <= 2 * pi + 1e-6
        p_epsilon = 1 / (2 * pi) * (exp ** (self.epsilon / 2))
        # sample from \mechanism(\pi, \epsilon)
        tmp = random.uniform(0, 1)
        if tmp < p_epsilon * (r - l):
            sampled = random.uniform(l, r) + self.private_val
        else:
            sampled = random.uniform(0, l) if random.uniform(0, 1) < 0.5 else random.uniform(r, 2 * pi)
        return sampled % (2 * pi)

    def linear_perturbation(self) -> "float":
        """
        Distance perturbation defined in the paper (Definition 3.2)
        :return: the perturbed distance
        """
        assert 0 <= self.private_val <= 1
        C = (exp ** (self.epsilon / 2) - 1) / (2 * exp ** self.epsilon - 2)
        assert 0 < C
        tmp = random.uniform(0, 1)
        if tmp < 2 * C:
            if self.private_val



    def square_wave_mechanism(self):
        """
        the square wave mechanism
        :return:
        """



class DiscreteMechanism:
    def __init__(self, private_val, epsilon, vals_domain: int):
        self.private_val = private_val
        self.epsilon = epsilon
        self.vals_domain = vals_domain

    def kRR(self, value, vals, epsilon):
        """
        the k-random response
        :param value: current value
        :param vals: the possible value
        :param epsilon: privacy budget
        :return:
        """
        values = vals.copy()
        p = np.e ** epsilon / (np.e ** epsilon + len(values) - 1)
        if np.random.random() < p:
            return value
        values.remove(value)
        return values[np.random.randint(low=0, high=len(values))]

    def exponential_mechanism(self):
        """
        the exponential mechanism
        :return:
        """
        return self.kRR(self.private_val, self.vals_domain, self.epsilon)







def square_wave_mechanism(poi_idx, R, epsilon):
    delta = max(poi_distance_matrix[poi_idx])
    t = R / delta
    b = (epsilon * (np.e ** (epsilon)) - np.e ** epsilon + 1) / (
                2 * np.e ** (epsilon) * (np.e ** epsilon - 1 - epsilon))
    x = random.uniform(0, 1)
    if x < (2 * b * np.e ** epsilon) / (2 * b * np.e ** epsilon + 1):
        perturbed_t = random.uniform(- b + t, b + t)
    else:
        x_1 = random.uniform(0, 1)
        if x_1 <= t:
            perturbed_t = random.uniform(- b, - b + t)
        else:
            perturbed_t = random.uniform(b + t, b + 1)
    perturbed_t = (perturbed_t + b) / (2 * b + 1)
    perturbed_t = perturbed_t * delta

    return perturbed_t
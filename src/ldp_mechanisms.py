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
            sampled = random.uniform(l, r) - pi + self.private_val
        else:
            sampled = random.uniform(0, l) if random.uniform(0, 1) < 0.5 else random.uniform(r, 2 * pi)
            sampled = sampled - pi + self.private_val
        return sampled % (2 * pi)

    def linear_perturbation(self) -> "float":
        """
        Distance perturbation defined in the paper (Definition 3.2)
        :return: the perturbed distance
        """
        assert 0 <= self.private_val <= 1 + 1e-6
        C = (exp ** (self.epsilon / 2) - 1) / (2 * exp ** self.epsilon - 2)
        assert 0 < C
        p = exp ** (self.epsilon / 2)
        tmp = random.uniform(0, 1)
        if tmp < 2 * C * p:
            if C <= self.private_val <= 1 - C:
                sampled = random.uniform(self.private_val - C, self.private_val + C)
            elif self.private_val < C:
                sampled = random.uniform(0, 2 * C)
            else:
                sampled = random.uniform(1 - 2 * C, 1)
        else:
            if C <= self.private_val <= 1 - C:
                left_proportion = (self.private_val - C) / (1 - 2 * C)
                sampled = random.uniform(0, self.private_val - C) if random.uniform(0, 1) <= left_proportion else random.uniform(self.private_val + C, 1)
            elif self.private_val < C:
                sampled = random.uniform(2 * C, 1)
            else:
                sampled = random.uniform(0, 1 - 2 * C)
        return sampled

    def square_wave_mechanism(self):
        """
        the square wave mechanism
        """
        pass


class DiscreteMechanism:
    def __init__(self, private_val, epsilon, total_num: int):
        self.private_val = private_val
        self.epsilon = epsilon
        self.total_num = total_num

    def krr(self) -> "int":
        """
        the k-RR mechanism
        """
        assert 0 <= self.private_val < self.total_num
        p = 1 / ((self.total_num - 1) + (exp ** self.epsilon))
        tmp = random.uniform(0, 1)
        if tmp < p:
            return random.randint(0, self.total_num)
        else:
            return self.private_val

    def exponential_mechanism(self):
        """
        the exponential mechanism
        """
        pass


# if __name__ == "__main__":
#     mechanism = PiecewiseMechanism(0, 1)
#     print(mechanism.linear_perturbation())

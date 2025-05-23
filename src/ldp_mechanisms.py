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

    def sw_circular(self) -> "float":
        """
        the redesigned square wave mechanism for circular perturbation
        """
        assert 0 <= self.private_val <= 2 * pi + 1e-6
        p = (exp ** self.epsilon - 1) / self.epsilon / (2 * pi)
        s = (exp ** self.epsilon * (self.epsilon - 1) + 1) / (exp ** self.epsilon - 1) ** 2
        # insight: sample from \mechanism(\pi, \epsilon) and then add self.private_val to the sample
        l = pi * (1 - s)
        r = pi * (1 + s)
        tmp = random.uniform(0, 1)
        if tmp < p * 2 * s:
            sampled = random.uniform(l, r) - pi + self.private_val
        else:
            sampled = random.uniform(0, l) if random.uniform(0, 1) < 0.5 else random.uniform(r, 2 * pi)
            sampled = sampled - pi + self.private_val
        return sampled % (2 * pi)

    def sw_linear(self) -> "float":
        """
        the redesigned square wave mechanism for linear perturbation
        """
        assert 0 <= self.private_val <= 1 + 1e-6
        # b = (self.epsilon * exp ** self.epsilon - exp ** self.epsilon + 1) / (
        #             2 * exp ** self.epsilon * (exp ** self.epsilon - 1 - self.epsilon))
        # assert 0 < b
        # p = exp ** self.epsilon / (2 * b * exp ** self.epsilon + 1)
        # # compress to [0, 1]
        # p = p * (1 + 2 * b)
        # assert abs(p - (exp ** self.epsilon - 1) / self.epsilon) < 1e-3
        p = (exp ** self.epsilon - 1) / self.epsilon
        C = (exp ** self.epsilon * (self.epsilon - 1) + 1) / (2 * (exp ** self.epsilon - 1) ** 2)
        tmp = random.uniform(0, 1)
        if tmp < p * 2 * C:
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
            while True:
                sampled = random.randint(0, self.total_num)
                if sampled != self.private_val:
                    return sampled
        else:
            return self.private_val

    def exp_mechanism_loc(self, location_list: "list"):
        """
        the exponential mechanism for location perturbation
        """
        length = len(location_list)
        # corner case: only one location -> sensitivity is inf
        if length == 1:
            return location_list[0]
        # score function array
        score_array = np.zeros(length)
        for i in range(length):
            score_array[i] = -np.linalg.norm(np.array(location_list[i]) - np.array(self.private_val))
        sensitivity = max(score_array) - min(score_array)
        # probability array
        p = np.zeros(length)
        for i in range(length):
            p[i] = exp ** (self.epsilon * score_array[i] / (2 * sensitivity))
        p = p / sum(p)
        assert abs(sum(p) - 1) < 1e-3
        # sample
        tmp = random.uniform(0, 1)
        index_perturbed = None
        for i in range(length):
            if tmp <= sum(p[:i + 1]):
                index_perturbed = i
                break
        return location_list[index_perturbed]

    def srr_3_groups(self, groups: "list"):
        """
        the SRR mechanism with 3 groups
        """
        assert len(groups) == 3
        length = len(groups[0]) + len(groups[1]) + len(groups[2])
        a_min = 2 / (2 * length * self.epsilon - (self.epsilon - 1) * (len(groups[1]) + 2 * len(groups[2])))
        delta_x = a_min * (self.epsilon - 1) / 2
        # probability of each point
        p_single = [a_min + 2 * delta_x, a_min + delta_x, a_min]
        # probability of each group
        p = [p_single[0] * len(groups[0]), p_single[1] * len(groups[1]), p_single[2] * len(groups[2])]
        # probability normalization constraint
        assert abs(sum(p) - 1) < 1e-2
        # sample
        tmp = random.uniform(0, 1)
        index_perturbed = None
        for i in range(3):
            if tmp <= sum(p[:i + 1]):
                index_perturbed = i
                break
        # sample from the group
        return random.choice(groups[index_perturbed])


# if __name__ == "__main__":
#     mechanism = PiecewiseMechanism(0, 1)
#     print(mechanism.linear_perturbation())

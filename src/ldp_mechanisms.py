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
        # If epsilon is too large, output the private value to avoid numerical error
        if self.epsilon > 10:
            return self.private_val
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
        if length == 0:
            raise ValueError("location_list must not be empty")
        # corner case: only one location -> deterministic output
        if length == 1:
            return location_list[0]

        # score function array
        score_array = np.zeros(length)
        private_val_array = np.array(self.private_val)
        for i in range(length):
            score_array[i] = -np.linalg.norm(np.array(location_list[i]) - private_val_array)
        sensitivity = float(np.ptp(score_array))

        # Degenerate geometry: all candidates have (almost) identical score.
        if (not np.isfinite(sensitivity)) or sensitivity <= 1e-12:
            p = np.full(length, 1.0 / length)
        else:
            # Stable softmax: shift by max(score) so all logits are <= 0.
            logits = self.epsilon * (score_array - np.max(score_array)) / (2.0 * sensitivity)
            weights = np.exp(logits)
            weight_sum = float(np.sum(weights))
            if (not np.isfinite(weight_sum)) or weight_sum <= 0:
                p = np.full(length, 1.0 / length)
            else:
                p = weights / weight_sum
        # sample
        tmp = random.uniform(0, 1)
        cdf = np.cumsum(p)
        # !!!
        # Optimization: use searchsorted to find the index of the sampled location in O(log(length)) time
        index_perturbed = int(np.searchsorted(cdf, tmp, side="right"))
        if index_perturbed >= length:
            index_perturbed = length - 1
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
        cdf = np.cumsum(p)
        index_perturbed = int(np.searchsorted(cdf, tmp, side="right"))
        if index_perturbed >= 3:
            index_perturbed = 2
        # sample from the group
        res = random.choice(groups[index_perturbed])
        return res

# if __name__ == "__main__":
#     mechanism = PiecewiseMechanism(0, 1)
#     print(mechanism.linear_perturbation())

import numpy as np
import random

def kRR(value, vals, epsilon):
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
import numpy as np
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.ldp_mechanisms import PiecewiseMechanism, DiscreteMechanism

pi = np.pi
x_max, y_max = 1, 1


def direction_tracs_vs_strawman(epsilon):
    """
    comparison of TRACS-D and Strawman on a random trajectory with n_points points
    """
    private_direction = np.random.uniform(0, 2 * pi)
    tracs_direction = PiecewiseMechanism(private_direction, epsilon).circular_perturbation()
    error_tracs_d = min(abs(tracs_direction - private_direction), abs(2 * pi - tracs_direction - private_direction))
    # strawman
    # 3-RR
    private_sector = private_direction // (2 * pi / 3)
    assert 0 <= private_sector <= 2
    direction_3rr = DiscreteMechanism(private_sector, epsilon, 3).krr()
    perturbed_direction = direction_3rr * (2 * pi / 3) + np.random.uniform(0, 1) * (2 * pi / 3)
    error_strawman_3rr = min(abs(perturbed_direction - private_direction), abs(2 * pi - perturbed_direction - private_direction))
    # 6-RR
    private_sector = private_direction // (pi / 3)
    assert 0 <= private_sector <= 5
    direction_6rr = DiscreteMechanism(private_sector, epsilon, 6).krr()
    perturbed_direction = direction_6rr * (pi / 3) + np.random.uniform(0, 1) * (pi / 3)
    error_strawman_6rr = min(abs(perturbed_direction - private_direction), abs(2 * pi - perturbed_direction - private_direction))
    # 12-RR
    private_sector = private_direction // (pi / 6)
    assert 0 <= private_sector <= 11
    direction_12rr = DiscreteMechanism(private_sector, epsilon, 12).krr()
    perturbed_direction = direction_12rr * (pi / 6) + np.random.uniform(0, 1) * (pi / 6)
    error_strawman_12rr = min(abs(perturbed_direction - private_direction), abs(2 * pi - perturbed_direction - private_direction))
    return error_tracs_d, error_strawman_3rr, error_strawman_6rr, error_strawman_12rr


if __name__ == '__main__':
    rows = []
    for epsilon in [2, 4, 6, 8, 10]:
        error_tracs_d, error_strawman_6rr, error_strawman_3rr, error_strawman_12rr = 0, 0, 0, 0
        for _ in range(10000):
            # average error
            error_tracs_d += direction_tracs_vs_strawman(epsilon)[0]
            error_strawman_3rr += direction_tracs_vs_strawman(epsilon)[1]
            error_strawman_6rr += direction_tracs_vs_strawman(epsilon)[2]
            error_strawman_12rr += direction_tracs_vs_strawman(epsilon)[3]
        row = {
            "epsilon": epsilon,
            "tracs_d": error_tracs_d / 10000,
            "strawman_3rr": error_strawman_3rr / 10000,
            "strawman_6rr": error_strawman_6rr / 10000,
            "strawman_12rr": error_strawman_12rr / 10000,
        }
        rows.append(row)
        print(f"epsilon: {epsilon}, tracs-d: {row['tracs_d']}, 3-RR: {row['strawman_3rr']}, 6-RR: {row['strawman_6rr']}, 12-RR: {row['strawman_12rr']}")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5,5))
    plt.rcParams["font.size"] = 20
    plt.plot([row["epsilon"] for row in rows], [row["strawman_3rr"] for row in rows], label="3-RR", linestyle=":", color="blue", marker="s")
    plt.plot([row["epsilon"] for row in rows], [row["strawman_6rr"] for row in rows], label="6-RR", linestyle="--", color="black", marker="x")
    plt.plot([row["epsilon"] for row in rows], [row["strawman_12rr"] for row in rows], label="12-RR", linestyle="--", color="green", marker="^")
    plt.plot([row["epsilon"] for row in rows], [row["tracs_d"] for row in rows], label="TraCS-D", linestyle="-.", color="red", marker="o")
    plt.xlabel(r'Privacy parameter $\varepsilon$')
    plt.ylabel(r"Error of the direction")
    plt.legend()
    plt.title("Figure 6")
    plt.show()

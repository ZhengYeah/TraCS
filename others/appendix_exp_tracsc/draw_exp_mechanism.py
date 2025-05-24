import numpy as np
import matplotlib.pyplot as plt
from src.utilities.discrete_location_space import discrete_location_grid
from src.ldp_mechanisms import DiscreteMechanism

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = 'serif'
plt.figure(figsize=(5, 5))

epsilon = 4
location_space = discrete_location_grid(10, x_max=1, y_max=1)
input_x = (0, 0)
perturbed_loc = DiscreteMechanism(input_x, epsilon, len(location_space)).exp_mechanism_loc(location_space)
# draw the location space
plt.scatter(*zip(*location_space), c='gray')
plt.scatter(*zip(*[input_x]), c='black')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xticks(np.arange(0, 1.1, 0.2))
plt.yticks(np.arange(0, 1.1, 0.2))
plt.savefig('appendix_exp_mechanism_1.pdf', bbox_inches='tight')
plt.show()
